"""少量真实模型联调；显式 --live 才调用已配置的 DeepSeek API。

运行：python -m evaluation.chat_smoke --live
与离线 unittest 不同，这个脚本验证真实自然语言提取和多轮承接。
"""
import argparse

from app.chat import ChatAssistant
from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from app.retrieval import create_llm


class NoKnowledgeCall:
    def invoke(self, value):
        raise AssertionError("算价场景不应进入知识生成链路")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="确认调用模型 API（产生少量费用）")
    args = parser.parse_args()
    if not args.live:
        parser.error("真实联调需添加 --live；离线测试请运行 python -m unittest discover -s tests")
    if not DEEPSEEK_API_KEY:
        parser.error("未配置 DEEPSEEK_API_KEY")

    llm = create_llm(DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL)
    assistant = ChatAssistant(llm, NoKnowledgeCall(), NoKnowledgeCall())
    cases = [
        (True, "一只中型犬，全程春节档期内住七天多少钱？", "quoted", "966.42"),
        (True, "我家狗寄养七天多少钱？", "needs_information", None),
        (False, "中型犬", "needs_information", None),
        (False, "普通时段，不是春节国庆", "quoted", "743.40"),
        (False, "那改成十天呢？", "quoted", "1062.00"),
        (True, "一只中型犬国庆前后住十天，其中三天是国庆，合计多少钱？", "unsupported", None),
        (True, "两只猫住同一个笼子，普通时段七天多少钱？", "unsupported", None),
        (True, "一只猫豪华间五一住三天多少钱？", "quoted", "294.00"),
    ]
    for reset, question, status, total in cases:
        if reset:
            assistant.clear()
        print(f"用户：{question}", flush=True)
        reply = assistant.reply(question)
        print(f"助手：{reply.answer}", flush=True)
        result = reply.trace[-1]
        assert result["status"] == status, reply.trace
        if total:
            assert result["quote"]["total"] == total, reply.trace
        print(f"PASS: {status}" + (f", total={total}" if total else ""), flush=True)
    print(f"真实模型联调通过：{len(cases)} 轮。有限样例通过不代表所有表达都能正确识别。")


if __name__ == "__main__":
    main()
