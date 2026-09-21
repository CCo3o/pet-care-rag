"""真实模型档期联调：--live 会调用 API，订单只写临时数据库。"""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from app.bookings import BookingStore
from app.chat import ChatAssistant
from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from app.retrieval import create_llm


class NoKnowledgeCall:
    def invoke(self, value):
        raise AssertionError("档期场景不应进入知识生成链路")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live or not DEEPSEEK_API_KEY:
        parser.error("需配置 API Key 并添加 --live；会产生少量模型调用费用")
    llm = create_llm(DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL)
    with TemporaryDirectory(prefix="booking-smoke-") as folder:
        def client():
            return ChatAssistant(llm, NoKnowledgeCall(), NoKnowledgeCall(),
                                 booking_store=BookingStore(Path(folder) / "bookings.sqlite3"))

        def ask(assistant, question, expected):
            print(f"用户：{question}", flush=True)
            reply = assistant.reply(question)
            print(reply.answer, flush=True)
            result = reply.trace[-1]
            assert result["status"] == expected, reply.trace
            return result

        a, b = client(), client()
        question = "一只猫标准间，2026年10月1日至4日都有位置吗？"
        result = ask(b, question, "availability")
        assert all(day["occupied"] == 0 for day in result["availability"]["days"])
        ask(a, "我要预订一只猫的标准间，2026年10月1日至4日。", "awaiting_confirmation")
        result = ask(b, question, "availability")
        assert all(day["occupied"] == 0 for day in result["availability"]["days"])
        ask(a, "确认预订", "booked")
        ask(a, "确认预订", "already_booked")
        result = ask(b, question, "availability")
        assert all(day["occupied"] == 1 for day in result["availability"]["days"])
        b.clear()
        b = client()
        result = ask(b, "一只猫标准间，2026年10月4日至5日还有位置吗？", "availability")
        assert [day["occupied"] for day in result["availability"]["days"]] == [1, 0]
        ask(client(), "一只猫标准间1到4号有位置吗？", "needs_information")
    print("PASS：真实模型查位、草案不占位、确认保存、跨会话更新、重复确认、结束日边界及缺参追问。未写入演示订单库。")


if __name__ == "__main__":
    main()
