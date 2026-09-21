"""确定性算价工具：把价格规则从大模型回答中拿出来，由代码精确计算。

示例：
    python -m app.pricing --pet-type 狗 --dog-size 中型 --days 10 --holiday 春节

这个模块目前只计算基础寄养费、节假日上浮和长住优惠；洗澡、接送等增值服务
将在业务工具的下一轮迭代中单独加入。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from app.config import BASE_DIR


PRICING_RULES_PATH = BASE_DIR / "data" / "pricing_rules.json"
MONEY_QUANTUM = Decimal("0.01")

PET_TYPE_ALIASES = {
    "猫": "cat", "cat": "cat",
    "狗": "dog", "犬": "dog", "dog": "dog",
    "异宠": "exotic", "exotic": "exotic",
}
DOG_SIZE_ALIASES = {
    "小型": "small", "小型犬": "small", "small": "small",
    "中型": "medium", "中型犬": "medium", "medium": "medium",
    "大型": "large", "大型犬": "large", "large": "large",
}
CAT_ROOM_ALIASES = {
    "标准": "standard", "标准间": "standard", "standard": "standard",
    "豪华": "deluxe", "豪华套间": "deluxe", "deluxe": "deluxe",
}
NO_HOLIDAY_ALIASES = {None, "", "无", "非节假日", "none", "null"}


class PricingInputError(ValueError):
    """用户输入无法套用现有价格规则时抛出。"""


@dataclass(frozen=True)
class PriceQuote:
    """一次基础寄养报价，以及可直接展示给顾客的计算过程。"""

    pet_label: str
    days: int
    daily_rate: Decimal
    holiday: str | None
    long_stay_discount_label: str | None
    base_subtotal: Decimal
    holiday_adjustment: Decimal
    long_stay_discount_amount: Decimal
    total: Decimal

    @staticmethod
    def format_money(amount: Decimal) -> str:
        return f"¥{amount.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP):.2f}"

    def calculation_lines(self) -> list[str]:
        """返回前端或客服可展示的、逐步可核对的算价过程。"""
        lines = [
            f"基础寄养费：{self.pet_label} {self.format_money(self.daily_rate)}/天 × {self.days} 天 = "
            f"{self.format_money(self.base_subtotal)}"
        ]
        if self.holiday:
            lines.append(
                f"{self.holiday}上浮：+{self.format_money(self.holiday_adjustment)}"
            )
        if self.long_stay_discount_label:
            lines.append(
                f"长住优惠（{self.long_stay_discount_label}）：-{self.format_money(self.long_stay_discount_amount)}"
            )
        lines.append(f"基础寄养总价：{self.format_money(self.total)}")
        return lines

    def to_dict(self) -> dict:
        """预留给未来 FastAPI / function calling 的稳定 JSON 结构。"""
        return {
            "pet_label": self.pet_label,
            "days": self.days,
            "daily_rate": str(self.daily_rate),
            "holiday": self.holiday,
            "long_stay_discount_label": self.long_stay_discount_label,
            "base_subtotal": str(self.base_subtotal),
            "holiday_adjustment": str(self.holiday_adjustment),
            "long_stay_discount_amount": str(self.long_stay_discount_amount),
            "total": str(self.total),
            "currency": "CNY",
            "calculation_lines": self.calculation_lines(),
        }


def load_pricing_rules(path: Path = PRICING_RULES_PATH) -> dict:
    """读取机器可执行的价格规则，集中维护，不在代码里硬编码价格。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"未找到价格规则文件：{path}") from exc


def _normalize(value: str | None, aliases: dict[str, str], field_name: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized not in aliases:
        choices = "、".join(sorted(set(aliases.values())))
        raise PricingInputError(f"{field_name}不支持：{value!r}。可选值：{choices}")
    return aliases[normalized]


def _normalize_holiday(holiday: str | None, rules: dict) -> str | None:
    normalized = (holiday or "").strip()
    if holiday in NO_HOLIDAY_ALIASES or normalized.lower() in NO_HOLIDAY_ALIASES:
        return None
    if normalized not in rules["holidays"]:
        choices = "、".join(rules["holidays"])
        raise PricingInputError(f"暂不支持的节假日：{holiday!r}。可选值：无、{choices}")
    return normalized


def _money(value: Decimal) -> Decimal:
    return value.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def calculate_price(
    *,
    pet_type: str,
    days: int,
    dog_size: str | None = None,
    cat_room: str = "标准",
    holiday: str | None = None,
) -> PriceQuote:
    """计算基础寄养报价。

    参数使用中文或英文别名均可，例如 ``pet_type='狗'``、``dog_size='中型'``。
    节假日先对基础寄养费上浮，再应用满足条件的最高长住优惠；增值服务不在本工具范围内。
    """
    if not isinstance(days, int) or isinstance(days, bool) or days <= 0:
        raise PricingInputError("寄养天数必须是大于 0 的整数")

    rules = load_pricing_rules()
    normalized_pet_type = _normalize(pet_type, PET_TYPE_ALIASES, "宠物类型")

    if normalized_pet_type == "dog":
        if not dog_size:
            raise PricingInputError("狗狗报价必须提供体型：小型、中型或大型")
        rate_key = _normalize(dog_size, DOG_SIZE_ALIASES, "狗狗体型")
    elif normalized_pet_type == "cat":
        rate_key = _normalize(cat_room, CAT_ROOM_ALIASES, "猫咪房型")
    else:
        rate_key = "standard"

    rate_rule = rules["base_daily_rates"][normalized_pet_type][rate_key]
    daily_rate = Decimal(str(rate_rule["amount"]))
    base_subtotal = daily_rate * days

    normalized_holiday = _normalize_holiday(holiday, rules)
    holiday_adjustment = Decimal("0")
    subtotal_after_holiday = base_subtotal
    if normalized_holiday:
        multiplier = Decimal(rules["holidays"][normalized_holiday]["base_rate_multiplier"])
        subtotal_after_holiday = base_subtotal * multiplier
        holiday_adjustment = subtotal_after_holiday - base_subtotal

    discount_rule = next(
        (rule for rule in rules["long_stay_discounts"] if days >= rule["minimum_days"]),
        None,
    )
    long_stay_discount_amount = Decimal("0")
    total = subtotal_after_holiday
    if discount_rule:
        multiplier = Decimal(discount_rule["multiplier"])
        total = subtotal_after_holiday * multiplier
        long_stay_discount_amount = subtotal_after_holiday - total

    return PriceQuote(
        pet_label=rate_rule["label"],
        days=days,
        daily_rate=_money(daily_rate),
        holiday=normalized_holiday,
        long_stay_discount_label=discount_rule["label"] if discount_rule else None,
        base_subtotal=_money(base_subtotal),
        holiday_adjustment=_money(holiday_adjustment),
        long_stay_discount_amount=_money(long_stay_discount_amount),
        total=_money(total),
    )


def main():
    parser = argparse.ArgumentParser(description="宠物寄养基础费用计算器（不调用大模型）")
    parser.add_argument("--pet-type", required=True, help="猫、狗或异宠")
    parser.add_argument("--days", required=True, type=int, help="寄养天数")
    parser.add_argument("--dog-size", help="狗狗体型：小型、中型或大型")
    parser.add_argument("--cat-room", default="标准", help="猫咪房型：标准或豪华，默认标准")
    parser.add_argument("--holiday", default="无", help="无、春节或国庆，默认无")
    args = parser.parse_args()

    try:
        quote = calculate_price(
            pet_type=args.pet_type,
            days=args.days,
            dog_size=args.dog_size,
            cat_room=args.cat_room,
            holiday=args.holiday,
        )
    except PricingInputError as exc:
        parser.error(str(exc))

    print("🐱🐶 宠物寄养基础报价")
    for line in quote.calculation_lines():
        print(f"   {line}")


if __name__ == "__main__":
    main()
