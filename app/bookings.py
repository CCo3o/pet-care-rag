"""共享模拟订单：每次读取 SQLite，确认预订时在同一事务中检查并占位。

入住范围包含起止日。例如 10 月 1 日至 4 日占用 1、2、3、4 日，5 日释放。
数据库按房型汇总位置数，一笔订单占一个位置；不存顾客姓名或联系方式。
"""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parent.parent
BOOKINGS_DB_PATH = BASE_DIR / "data" / "bookings.sqlite3"
CAPACITY_PATH = BASE_DIR / "data" / "booking_capacity.json"
# 防止错误参数生成极大日期数组，这是演示工具输入限制，不是门店业务政策。
MAX_QUERY_DAYS = 366
ROOM_LABELS = {
    "cat_standard": "猫标准单间",
    "cat_deluxe": "猫豪华套间",
    "dog_small": "小型犬标准间",
    "dog_medium": "中型犬标准间",
    "dog_large": "大型犬标准间",
    "exotic": "异宠寄养位",
}


class BookingInputError(ValueError):
    """房型、日期或请求编号不符合预订工具输入要求。"""


class BookingStore:
    def __init__(
        self,
        db_path: Path | str | None = None,
        capacity_path: Path | str | None = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else BOOKINGS_DB_PATH
        self.capacity_path = (
            Path(capacity_path) if capacity_path is not None else CAPACITY_PATH
        )
        self._load_capacities()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS bookings (
                        booking_id TEXT PRIMARY KEY,
                        request_id TEXT NOT NULL UNIQUE,
                        room_type TEXT NOT NULL,
                        start_date TEXT NOT NULL,
                        end_date TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        status TEXT NOT NULL DEFAULT 'active',
                        cancelled_at TEXT,
                        CHECK (start_date <= end_date)
                    )"""
                )
                columns = {row["name"] for row in conn.execute("PRAGMA table_info(bookings)")}
                if "status" not in columns:
                    conn.execute("ALTER TABLE bookings ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
                if "cancelled_at" not in columns:
                    conn.execute("ALTER TABLE bookings ADD COLUMN cancelled_at TEXT")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS bookings_room_dates "
                    "ON bookings (room_type, start_date, end_date)"
                )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_capacities(self) -> dict[str, int]:
        with self.capacity_path.open(encoding="utf-8") as file:
            config = json.load(file)
        capacities = config.get("capacities")
        if not isinstance(capacities, dict) or set(capacities) != set(ROOM_LABELS):
            raise BookingInputError("容量配置必须包含所有已支持房型。")
        if any(type(value) is not int or value < 0 for value in capacities.values()):
            raise BookingInputError("房型容量必须为非负整数。")
        return capacities

    def room_types(self) -> list[dict]:
        capacities = self._load_capacities()
        return [
            {"room_type": key, "label": label, "capacity": capacities[key]}
            for key, label in ROOM_LABELS.items()
        ]

    @staticmethod
    def _validate(room_type: str, start_date: str, end_date: str) -> list[str]:
        if not isinstance(room_type, str) or room_type not in ROOM_LABELS:
            raise BookingInputError("请选择支持的宠物房型。")
        dates = []
        for name, value in (("开始日期", start_date), ("结束日期", end_date)):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
                raise BookingInputError(f"{name}必须为完整的 YYYY-MM-DD 日期。")
            try:
                dates.append(date.fromisoformat(value))
            except ValueError as exc:
                raise BookingInputError(f"{name}不是有效日期。") from exc
        first, last = dates
        if last < first:
            raise BookingInputError("结束日期不能早于开始日期。")
        length = (last - first).days + 1
        if length > MAX_QUERY_DAYS:
            raise BookingInputError(f"演示工具单次最多查询或预订 {MAX_QUERY_DAYS} 天。")
        return [(first + timedelta(days=offset)).isoformat() for offset in range(length)]

    def _availability(
        self, conn: sqlite3.Connection, room_type: str, dates: list[str]
    ) -> dict:
        capacity = self._load_capacities()[room_type]
        rows = conn.execute(
            "SELECT start_date, end_date FROM bookings "
            "WHERE room_type = ? AND status = 'active' AND start_date <= ? AND end_date >= ?",
            (room_type, dates[-1], dates[0]),
        ).fetchall()
        # 每条相交订单贡献一个闭区间；差分数组避免逐订单逐日反复遍历。
        offsets = {day: index for index, day in enumerate(dates)}
        changes = [0] * (len(dates) + 1)
        for row in rows:
            first = max(row["start_date"], dates[0])
            last = min(row["end_date"], dates[-1])
            changes[offsets[first]] += 1
            changes[offsets[last] + 1] -= 1
        days = []
        occupied = 0
        for index, day in enumerate(dates):
            occupied += changes[index]
            days.append({"date": day, "occupied": occupied, "remaining": max(0, capacity - occupied)})
        available = all(day["remaining"] > 0 for day in days)
        status = "available" if available else (
            "partial" if any(day["remaining"] > 0 for day in days) else "full"
        )
        return {
            "room_type": room_type,
            "label": ROOM_LABELS[room_type],
            "start_date": dates[0],
            "end_date": dates[-1],
            "capacity": capacity,
            "days": days,
            "available": available,
            "status": status,
        }

    def check_availability(self, room_type: str, start_date: str, end_date: str) -> dict:
        """读取已提交订单的最新快照，仅返回逐日汇总，不返回其他顾客订单。"""
        dates = self._validate(room_type, start_date, end_date)
        with closing(self._connect()) as conn:
            return self._availability(conn, room_type, dates)

    @staticmethod
    def _booking_result(row: sqlite3.Row | dict, status: str) -> dict:
        return {
            "status": status,
            "booking_id": row["booking_id"],
            "room_type": row["room_type"],
            "label": ROOM_LABELS[row["room_type"]],
            "start_date": row["start_date"],
            "end_date": row["end_date"],
        }

    def create_booking(
        self, room_type: str, start_date: str, end_date: str, request_id: str
    ) -> dict:
        """同一写事务内重新检查容量并写入；重复请求不会重复占位。

        request_id 由应用服务端生成。同一个编号只能对应同一组房型和日期。
        本方法仅处理已经获得顾客明确确认的预订，确认流程由聊天层负责。
        """
        dates = self._validate(room_type, start_date, end_date)
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 128:
            raise BookingInputError("预订请求编号必须是 1 至 128 字符的非空字符串。")
        with closing(self._connect()) as conn:
            # SQLite 先取得写锁，再查询余位；其他进程必须等待本事务提交。
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM bookings WHERE request_id = ?", (request_id,)
            ).fetchone()
            if existing is not None:
                if (existing["room_type"], existing["start_date"], existing["end_date"]) != (
                    room_type, start_date, end_date
                ):
                    raise BookingInputError("相同请求编号不能用于不同预订参数。")
                conn.commit()
                return self._booking_result(existing, "already_booked")
            availability = self._availability(conn, room_type, dates)
            if not availability["available"]:
                conn.commit()
                return {"status": "unavailable", "availability": availability}
            booking = {
                "booking_id": uuid4().hex,
                "room_type": room_type,
                "start_date": start_date,
                "end_date": end_date,
            }
            conn.execute(
                "INSERT INTO bookings (booking_id, request_id, room_type, start_date, end_date) "
                "VALUES (?, ?, ?, ?, ?)",
                (booking["booking_id"], request_id, room_type, start_date, end_date),
            )
            conn.commit()
            return self._booking_result(booking, "booked")

    def cancel_booking(self, booking_id: str) -> dict:
        """标记已保存订单为取消，释放档期但保留记录以便追溯。"""
        if not isinstance(booking_id, str) or not re.fullmatch(r"[0-9a-fA-F]{32}", booking_id):
            raise BookingInputError("订单编号格式不正确，请提供完整的订单编号。")
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM bookings WHERE booking_id = ?", (booking_id,)).fetchone()
            if row is None:
                conn.commit()
                return {"status": "not_found", "booking_id": booking_id}
            if row["status"] == "cancelled":
                conn.commit()
                return self._booking_result(row, "already_cancelled")
            conn.execute(
                "UPDATE bookings SET status = 'cancelled', cancelled_at = CURRENT_TIMESTAMP "
                "WHERE booking_id = ? AND status = 'active'",
                (booking_id,),
            )
            conn.commit()
            return self._booking_result(row, "cancelled")
