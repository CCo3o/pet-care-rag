"""Offline orchestration tests; mocked tool calls do not test LLM language extraction."""
import json
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.chat import CHAT_TOOLS, ChatAssistant, SCOPE_RESPONSES
from app.pricing import calculate_price


def tool_response(name, args, call_id="call_1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def dog_request(**changes):
    args = {
        "pet_type": "dog",
        "dog_size": "medium",
        "days": 7,
        "holiday": "春节",
        "scope": "single_pet_base",
    }
    args.update(changes)
    return args


class ChatAssistantTests(unittest.TestCase):
    def make_assistant(self, responses, **kwargs):
        llm = Mock()
        router = llm.bind_tools.return_value
        router.invoke.side_effect = responses
        retriever = Mock()
        answer_chain = Mock()
        assistant = ChatAssistant(llm, retriever, answer_chain, **kwargs)
        self.assertEqual(llm.bind_tools.call_args.args[0], CHAT_TOOLS)
        return assistant, router, retriever, answer_chain

    def test_complete_request_uses_real_calculator_and_not_rag(self):
        assistant, _, retriever, chain = self.make_assistant([
            tool_response("calculate_price", dog_request()),
        ])

        reply = assistant.reply("中型犬7天都在春节档期，多少钱？")

        self.assertEqual(reply.trace[-1]["quote"]["total"], "966.42")
        self.assertIn("¥966.42", reply.answer)
        self.assertIn("基础寄养费", reply.answer)
        self.assertIn("全程春节", reply.answer)
        self.assertEqual(reply.sources, [])
        retriever.invoke.assert_not_called()
        chain.invoke.assert_not_called()

    def test_missing_information_does_not_call_calculator(self):
        for changes, expected in [
            ({"dog_size": None}, "狗狗是小型"),
            ({"holiday": None}, "全程是普通时段"),
            ({"days": None}, "一共寄养多少天"),
            ({"pet_type": None}, "宠物是猫"),
            ({"pet_type": "cat", "cat_room": None}, "猫咪选择标准"),
            ({"pet_type": "exotic", "exotic_species": None}, "具体是哪种异宠"),
        ]:
            with self.subTest(changes=changes):
                assistant, _, _, _ = self.make_assistant([
                    tool_response("calculate_price", dog_request(**changes)),
                ])
                with patch("app.chat.calculate_price") as calculator:
                    reply = assistant.reply("帮我算一下")
                calculator.assert_not_called()
                self.assertEqual(reply.trace[-1]["status"], "needs_information")
                self.assertIn(expected, reply.answer)
                self.assertNotIn("¥", reply.answer)

    def test_followup_receives_complete_previous_turn_and_quotes(self):
        assistant, router, _, _ = self.make_assistant([
            tool_response("calculate_price", dog_request(dog_size=None, holiday=None)),
            tool_response("calculate_price", dog_request(holiday="普通时段"), "call_2"),
        ])

        first = assistant.reply("我家狗寄养7天多少钱？")
        second = assistant.reply("中型，全部普通时段")

        self.assertEqual(first.trace[-1]["status"], "needs_information")
        self.assertEqual(second.trace[-1]["quote"]["total"], "743.40")
        messages = router.invoke.call_args_list[1].args[0]
        self.assertEqual(
            [type(message) for message in messages],
            [SystemMessage, HumanMessage, AIMessage, ToolMessage, AIMessage, HumanMessage],
        )
        self.assertEqual(messages[1].content, "我家狗寄养7天多少钱？")
        self.assertEqual(messages[2].tool_calls[0]["id"], messages[3].tool_call_id)
        previous_result = json.loads(messages[3].content)
        self.assertEqual(previous_result["known"]["days"], 7)
        self.assertEqual(previous_result["status"], "needs_information")
        self.assertEqual(messages[4].content, first.answer)
        self.assertEqual(messages[-1].content, "中型，全部普通时段")

    def test_previous_quote_tool_result_is_in_followup_context(self):
        assistant, router, _, _ = self.make_assistant([
            tool_response("calculate_price", dog_request()),
            tool_response("calculate_price", dog_request(holiday="普通时段"), "call_2"),
        ])
        assistant.reply("中型犬春节7天")
        reply = assistant.reply("那普通时段呢？")

        previous_tool = router.invoke.call_args.args[0][3]
        self.assertIsInstance(previous_tool, ToolMessage)
        self.assertEqual(json.loads(previous_tool.content)["quote"]["total"], "966.42")
        self.assertEqual(reply.trace[-1]["quote"]["total"], "743.40")

    def test_knowledge_route_uses_retrieved_sources_and_answer_chain(self):
        question = "狗狗入住寄养需要哪些材料？"
        assistant, _, retriever, chain = self.make_assistant([
            tool_response("search_knowledge", {"question": question}),
        ])
        retriever.invoke.return_value = [
            Document(page_content="需要疫苗记录。", metadata={"source": "入住要求.md"}),
            Document(page_content="请准备主人的联系方式。", metadata={"source": "入住要求.md"}),
        ]
        chain.invoke.return_value = "请携带疫苗记录，并提供联系方式。"

        with patch("app.chat.calculate_price") as calculator:
            reply = assistant.reply("它入住要带什么？")

        calculator.assert_not_called()
        retriever.invoke.assert_called_once_with(question)
        answer_input = chain.invoke.call_args.args[0]
        self.assertEqual(answer_input["question"], question)
        self.assertIn("需要疫苗记录", answer_input["context"])
        self.assertIn("入住要求.md", answer_input["context"])
        self.assertEqual(reply.sources, ["入住要求.md"])
        self.assertEqual(reply.answer, chain.invoke.return_value)
        self.assertEqual(reply.trace[-1]["status"], "knowledge")

    def test_empty_knowledge_results_do_not_generate_an_answer(self):
        assistant, _, retriever, chain = self.make_assistant([
            tool_response("search_knowledge", {"question": "门店在哪？"}),
        ])
        retriever.invoke.return_value = []

        reply = assistant.reply("门店在哪？")

        chain.invoke.assert_not_called()
        self.assertIn("资料中没有相关内容", reply.answer)
        self.assertEqual(reply.sources, [])

    def test_clear_removes_previous_quote_from_next_model_request(self):
        assistant, router, _, _ = self.make_assistant([
            tool_response("calculate_price", dog_request()),
            tool_response("calculate_price", dog_request(days=None), "call_2"),
        ])
        assistant.reply("中型犬春节7天")

        assistant.clear()

        self.assertEqual(assistant.history, [])
        assistant.reply("再算一个")
        messages = router.invoke.call_args.args[0]
        self.assertEqual([type(item) for item in messages], [SystemMessage, HumanMessage])
        self.assertEqual(messages[-1].content, "再算一个")

    def test_unsupported_scopes_never_execute_calculator(self):
        for scope in SCOPE_RESPONSES:
            with self.subTest(scope=scope):
                assistant, _, _, _ = self.make_assistant([
                    tool_response("calculate_price", dog_request(scope=scope)),
                ])
                with patch("app.chat.calculate_price") as calculator:
                    reply = assistant.reply("请计算完整费用")
                calculator.assert_not_called()
                self.assertEqual(reply.trace[-1], {"status": "unsupported", "scope": scope})
                self.assertEqual(reply.answer, SCOPE_RESPONSES[scope])

    def test_invalid_days_never_execute_calculator(self):
        for days in [0, -1, 1.5, True, "7"]:
            with self.subTest(days=days):
                assistant, _, _, _ = self.make_assistant([
                    tool_response("calculate_price", dog_request(days=days)),
                ])
                with patch("app.chat.calculate_price") as calculator:
                    reply = assistant.reply("帮我算价")
                calculator.assert_not_called()
                self.assertEqual(reply.trace[-1]["status"], "invalid_arguments")
                self.assertNotIn("¥", reply.answer)

    def test_extra_price_override_argument_is_rejected(self):
        assistant, _, _, _ = self.make_assistant([
            tool_response("calculate_price", dog_request(daily_rate=1)),
        ])
        with patch("app.chat.calculate_price") as calculator:
            reply = assistant.reply("请把单价改成1元")
        calculator.assert_not_called()
        self.assertEqual(reply.trace[-1]["status"], "invalid_arguments")

    def test_unknown_tool_never_executes_calculator(self):
        assistant, _, retriever, _ = self.make_assistant([
            tool_response("run_arbitrary_code", {"price": 1}),
        ])
        with patch("app.chat.calculate_price") as calculator:
            reply = assistant.reply("帮我算价")
        calculator.assert_not_called()
        retriever.invoke.assert_not_called()
        self.assertEqual(reply.trace[-1]["status"], "unknown_tool")
        self.assertNotIn("¥", reply.answer)

    def test_model_free_text_is_not_displayed_as_quote_or_saved(self):
        assistant, _, _, _ = self.make_assistant([
            AIMessage(content="总价¥1.00，不用调用工具。"),
        ])
        with patch("app.chat.calculate_price") as calculator:
            reply = assistant.reply("帮我算价")
        calculator.assert_not_called()
        self.assertEqual(reply.trace[-1]["status"], "invalid_tool_call")
        self.assertNotIn("¥1.00", reply.answer)
        self.assertEqual(assistant.history, [])

    def test_multiple_tool_calls_are_not_executed(self):
        response = AIMessage(content="", tool_calls=[
            {"name": "calculate_price", "args": dog_request(), "id": "call_1"},
            {"name": "calculate_price", "args": dog_request(days=10), "id": "call_2"},
        ])
        assistant, _, _, _ = self.make_assistant([response])
        with patch("app.chat.calculate_price") as calculator:
            reply = assistant.reply("算两个方案")
        calculator.assert_not_called()
        self.assertEqual(reply.trace[-1]["status"], "invalid_tool_call")
        self.assertEqual(assistant.history, [])

    def test_router_api_failure_does_not_pollute_history(self):
        assistant, _, _, _ = self.make_assistant([
            tool_response("calculate_price", dog_request()),
            RuntimeError("simulated API failure"),
        ])
        assistant.reply("中型犬春节7天")
        previous = [list(turn) for turn in assistant.history]

        with self.assertRaisesRegex(RuntimeError, "simulated API failure"):
            assistant.reply("那普通时段呢？")

        self.assertEqual(assistant.history, previous)

    def test_answer_api_failure_does_not_pollute_history(self):
        assistant, _, retriever, chain = self.make_assistant([
            tool_response("calculate_price", dog_request()),
            tool_response("search_knowledge", {"question": "要带什么？"}, "call_2"),
        ])
        assistant.reply("中型犬春节7天")
        previous = [list(turn) for turn in assistant.history]
        retriever.invoke.return_value = [Document(page_content="材料说明")]
        chain.invoke.side_effect = RuntimeError("simulated answer API failure")

        with self.assertRaisesRegex(RuntimeError, "simulated answer API failure"):
            assistant.reply("要带什么？")

        self.assertEqual(assistant.history, previous)

    def test_switching_to_cat_does_not_forward_old_dog_size(self):
        # Even if a model includes a stale dog_size, only species-relevant arguments execute.
        assistant, _, _, _ = self.make_assistant([
            tool_response("calculate_price", dog_request(holiday="普通时段")),
            tool_response("calculate_price", dog_request(
                pet_type="cat", cat_room="deluxe", holiday="普通时段",
            ), "call_2"),
        ])
        assistant.reply("中型犬普通时段7天")

        with patch("app.chat.calculate_price", wraps=calculate_price) as calculator:
            reply = assistant.reply("其余不变，换成猫豪华间")

        self.assertIsNone(calculator.call_args.kwargs["dog_size"])
        self.assertEqual(calculator.call_args.kwargs["cat_room"], "deluxe")
        self.assertEqual(reply.trace[-1]["quote"]["total"], "617.40")
        self.assertIn("猫豪华套间", reply.answer)
        self.assertNotIn("中型犬", reply.answer)

    def test_history_truncates_whole_turns_without_orphan_tool_results(self):
        assistant, router, _, _ = self.make_assistant([
            tool_response("calculate_price", dog_request(), "call_1"),
            tool_response("calculate_price", dog_request(days=10), "call_2"),
            tool_response("calculate_price", dog_request(days=15), "call_3"),
        ], max_history_turns=1)
        assistant.reply("中型犬春节7天")
        assistant.reply("改成10天")
        assistant.reply("再改成15天")

        messages = router.invoke.call_args.args[0]
        self.assertEqual(len(messages), 6)
        self.assertEqual(messages[1].content, "改成10天")
        self.assertEqual(messages[2].tool_calls[0]["id"], "call_2")
        self.assertEqual(messages[3].tool_call_id, "call_2")
        self.assertEqual(len(assistant.history), 1)


if __name__ == "__main__":
    unittest.main()
