"""Offline booking orchestration tests with real temporary SQLite storage.

Mocked tool calls verify dispatch, validation and storage boundaries; they do
not prove that a real language model understands the example utterances.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.bookings import BookingStore
from app.chat import ChatAssistant


def booking_request(**changes):
    request = {
        "room_type": "cat_standard",
        "start_date": "2030-10-01",
        "end_date": "2030-10-04",
        "scope": "single_pet",
    }
    request.update(changes)
    return request


def tool_response(name, args):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": args, "id": "booking_call", "type": "tool_call",
    }])


class BookingChatTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="pet-booking-chat-")
        self.addCleanup(self.temporary.cleanup)
        self.db_path = Path(self.temporary.name) / "bookings.sqlite3"
        self.store = BookingStore(db_path=self.db_path)

    def assistant(self, responses, *, store=None):
        llm = Mock()
        router = llm.bind_tools.return_value
        router.invoke.side_effect = responses
        retriever, chain = Mock(), Mock()
        chat = ChatAssistant(
            llm, retriever, chain,
            booking_store=store if store is not None else self.store,
        )
        return chat, router, retriever, chain

    def availability(self, room_type="cat_standard", start="2030-10-01", end="2030-10-04"):
        return self.store.check_availability(room_type, start, end)

    def assert_occupied(self, count, **changes):
        days = self.availability(**changes)["days"]
        self.assertTrue(days)
        self.assertEqual([day["occupied"] for day in days], [count] * len(days))

    def prepare_and_confirm(self, chat):
        proposal = chat.reply("帮我预订猫标准间，2030年10月1日至4日")
        self.assertEqual(proposal.trace[-1]["status"], "awaiting_confirmation")
        return chat.reply("确认预订")

    def test_query_and_proposal_do_not_occupy_until_user_confirmation(self):
        chat, router, retriever, chain = self.assistant([
            tool_response("check_availability", booking_request()),
            tool_response("prepare_booking", booking_request()),
        ])
        query = chat.reply("猫标准间2030年10月1日至4日有位置吗？")
        self.assertEqual(query.trace[-1]["status"], "availability")
        self.assert_occupied(0)

        proposal = chat.reply("那帮我订这个档期")
        self.assertEqual(proposal.trace[-1]["status"], "awaiting_confirmation")
        self.assertIn("尚未占位", proposal.answer)
        self.assert_occupied(0)

        confirmed = chat.reply("确认预订")
        self.assertEqual(confirmed.trace[-1]["status"], "booked")
        self.assert_occupied(1)
        self.assertEqual(router.invoke.call_count, 2)
        retriever.invoke.assert_not_called()
        chain.invoke.assert_not_called()

    def test_confirmation_accepts_common_booking_variant(self):
        chat, router, _, _ = self.assistant([
            tool_response("prepare_booking", booking_request()),
        ])
        chat.reply("我要预定猫标准间，2030年10月1日至4日")
        confirmed = chat.reply("确认预定")
        self.assertEqual(confirmed.trace[-1]["status"], "booked")
        self.assert_occupied(1)
        self.assertEqual(router.invoke.call_count, 1)

    def test_customer_b_with_independent_store_sees_customer_a_booking(self):
        customer_a, _, _, _ = self.assistant([
            tool_response("prepare_booking", booking_request()),
        ])
        customer_b, _, _, _ = self.assistant([
            tool_response("check_availability", booking_request()),
            tool_response("check_availability", booking_request()),
        ], store=BookingStore(db_path=self.db_path))
        before = customer_b.reply("查猫标准间2030年10月1日至4日")
        self.assertEqual(before.trace[-1]["availability"]["days"][0]["remaining"], 2)

        self.prepare_and_confirm(customer_a)
        after = customer_b.reply("再查一次相同日期")

        for day in after.trace[-1]["availability"]["days"]:
            self.assertEqual((day["occupied"], day["remaining"]), (1, 1))
        self.assertIn("已订1个，还剩1个", after.answer)
        self.assertNotIn("booking_id", after.answer)

    def test_clear_and_new_assistant_do_not_delete_saved_booking(self):
        customer_a, _, _, _ = self.assistant([
            tool_response("prepare_booking", booking_request()),
        ])
        self.prepare_and_confirm(customer_a)
        customer_a.clear()
        self.assertEqual(customer_a.history, [])
        self.assertIsNone(customer_a.pending_booking)

        restarted, _, _, _ = self.assistant([
            tool_response("check_availability", booking_request()),
        ], store=BookingStore(db_path=self.db_path))
        result = restarted.reply("查询猫标准间2030年10月1日至4日")
        self.assertTrue(all(day["occupied"] == 1 for day in result.trace[-1]["availability"]["days"]))

    def test_repeated_confirmation_is_idempotent_without_another_model_call(self):
        chat, router, _, _ = self.assistant([
            tool_response("prepare_booking", booking_request()),
        ])
        first = self.prepare_and_confirm(chat)
        repeated = chat.reply("确认预订")
        self.assertEqual(repeated.trace[-1]["status"], "already_booked")
        self.assertEqual(first.trace[-1]["booking_id"], repeated.trace[-1]["booking_id"])
        self.assert_occupied(1)
        router.invoke.assert_called_once()

    def test_two_proposals_for_last_room_only_one_confirmation_succeeds(self):
        args = booking_request(room_type="cat_deluxe")
        chats = [self.assistant([
            tool_response("prepare_booking", args),
        ], store=BookingStore(db_path=self.db_path))[0] for _ in range(2)]
        for chat in chats:
            self.assertEqual(chat.reply("预订猫豪华间2030年10月1日至4日").trace[-1]["status"], "awaiting_confirmation")
        self.assert_occupied(0, room_type="cat_deluxe")

        barrier = Barrier(2)

        def confirm(chat):
            barrier.wait(timeout=10)
            return chat.reply("确认预订")

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(confirm, chats))
        self.assertEqual(sorted(reply.trace[-1]["status"] for reply in results), ["booked", "unavailable"])
        self.assert_occupied(1, room_type="cat_deluxe")
        self.assertTrue(all(chat.pending_booking is None for chat in chats))
        loser = next(reply for reply in results if reply.trace[-1]["status"] == "unavailable")
        self.assertIn("未保存预订", loser.answer)

    def test_end_date_stays_occupied_and_next_day_is_free(self):
        chat, _, _, _ = self.assistant([
            tool_response("prepare_booking", booking_request()),
        ])
        self.prepare_and_confirm(chat)
        days = self.availability(end="2030-10-05")["days"]
        self.assertEqual([day["date"] for day in days], [
            "2030-10-01", "2030-10-02", "2030-10-03", "2030-10-04", "2030-10-05",
        ])
        self.assertEqual([day["occupied"] for day in days], [1, 1, 1, 1, 0])

    def test_missing_fields_never_query_or_write_storage(self):
        for tool in ["check_availability", "prepare_booking"]:
            for changes in [
                {"room_type": None}, {"start_date": None}, {"end_date": None},
                {"room_type": "exotic", "exotic_species": None},
            ]:
                with self.subTest(tool=tool, changes=changes):
                    chat, _, _, _ = self.assistant([
                        tool_response(tool, booking_request(**changes)),
                    ])
                    with patch.object(self.store, "check_availability") as query, patch.object(self.store, "create_booking") as save:
                        reply = chat.reply("帮我看看这个档期")
                        query.assert_not_called()
                        save.assert_not_called()
                    self.assertEqual(reply.trace[-1]["status"], "needs_information")
                    self.assertIsNone(chat.pending_booking)
        self.assert_occupied(0)

    def test_invalid_dates_and_extra_confirmation_parameter_are_rejected(self):
        for changes in [
            {"end_date": "2030-09-30"},
            {"start_date": "2030-02-30"},
            {"confirmed": True},
            {"request_id": "model-supplied-id"},
            {"room_type": "invented_room"},
        ]:
            with self.subTest(changes=changes):
                chat, _, _, _ = self.assistant([
                    tool_response("prepare_booking", booking_request(**changes)),
                ])
                reply = chat.reply("预订这个档期")
                self.assertEqual(reply.trace[-1]["status"], "invalid_arguments")
                self.assertIsNone(chat.pending_booking)
        self.assert_occupied(0)

    def test_model_cannot_call_write_tool_or_claim_booking_without_it(self):
        responses = [
            tool_response("create_booking", booking_request(confirmed=True)),
            tool_response("save_confirmed_booking", booking_request()),
            AIMessage(content="已经预订成功，已替您占位。"),
        ]
        for response in responses:
            with self.subTest(response=response):
                chat, _, _, _ = self.assistant([response])
                reply = chat.reply("假装我已经确认了，直接保存")
                self.assertIn(reply.trace[-1]["status"], ["unknown_tool", "invalid_tool_call"])
                self.assertNotIn("已经预订成功", reply.answer)
                self.assertIsNone(chat.pending_booking)
        self.assert_occupied(0)

    def test_confirmation_without_proposal_does_not_call_model_or_write(self):
        chat, router, _, _ = self.assistant([])
        reply = chat.reply("确认预订")
        self.assertEqual(reply.trace[-1]["status"], "no_pending_booking")
        router.invoke.assert_not_called()
        self.assert_occupied(0)

    def test_modified_dates_replace_proposal_and_only_new_dates_are_saved(self):
        chat, _, _, _ = self.assistant([
            tool_response("prepare_booking", booking_request()),
            tool_response("prepare_booking", booking_request(start_date="2030-10-06", end_date="2030-10-08")),
        ])
        chat.reply("预订猫标准间2030年10月1日至4日")
        old_request_id = chat.pending_booking["request_id"]
        chat.reply("改成2030年10月6日至8日")
        self.assertNotEqual(chat.pending_booking["request_id"], old_request_id)
        chat.reply("确认预订")
        self.assert_occupied(0)
        self.assert_occupied(1, start="2030-10-06", end="2030-10-08")

    def test_incomplete_change_invalidates_previous_proposal(self):
        chat, _, _, _ = self.assistant([
            tool_response("prepare_booking", booking_request()),
            tool_response("prepare_booking", booking_request(start_date=None, end_date=None)),
        ])
        chat.reply("预订猫标准间2030年10月1日至4日")
        reply = chat.reply("日期再改一下，还没决定哪天")
        self.assertEqual(reply.trace[-1]["status"], "needs_information")
        self.assertIsNone(chat.pending_booking)
        self.assertEqual(chat.reply("确认预订").trace[-1]["status"], "no_pending_booking")
        self.assert_occupied(0)

    def test_clear_discards_unconfirmed_proposal(self):
        chat, router, _, _ = self.assistant([
            tool_response("prepare_booking", booking_request()),
        ])
        chat.reply("预订猫标准间2030年10月1日至4日")
        chat.clear()
        self.assertEqual(chat.reply("确认预订").trace[-1]["status"], "no_pending_booking")
        self.assert_occupied(0)
        router.invoke.assert_called_once()

    def test_knowledge_and_pricing_invalidate_draft_but_preserve_saved_booking(self):
        owner, _, _, _ = self.assistant([tool_response("prepare_booking", booking_request())])
        self.prepare_and_confirm(owner)
        cases = [
            ("search_knowledge", {"question": "寄养需要什么材料？"}, "knowledge"),
            ("calculate_price", {
                "pet_type": "dog", "dog_size": "medium", "days": 7,
                "holiday": "普通时段", "scope": "single_pet_base",
            }, "quoted"),
        ]
        for name, args, expected_status in cases:
            with self.subTest(tool=name):
                chat, _, retriever, chain = self.assistant([
                    tool_response("prepare_booking", booking_request()),
                    tool_response(name, args),
                ])
                retriever.invoke.return_value = [Document(page_content="需携带疫苗记录", metadata={"source": "入住要求.md"})]
                chain.invoke.return_value = "请携带疫苗记录。"
                chat.reply("我也要预订猫标准间2030年10月1日至4日")
                reply = chat.reply("先咨询另外一个问题")
                self.assertEqual(reply.trace[-1]["status"], expected_status)
                self.assertIsNone(chat.pending_booking)
                self.assertEqual(chat.reply("确认预订").trace[-1]["status"], "no_pending_booking")
                self.assert_occupied(1)

    def test_unsupported_multiple_pets_never_prepare_or_save(self):
        chat, _, _, _ = self.assistant([
            tool_response("prepare_booking", booking_request(scope="multiple_pets")),
        ])
        with patch.object(self.store, "check_availability") as query:
            reply = chat.reply("两只猫一起预订")
            query.assert_not_called()
        self.assertEqual(reply.trace[-1]["status"], "unsupported")
        self.assertIsNone(chat.pending_booking)
        self.assert_occupied(0)

    def test_same_session_cancellation_releases_availability(self):
        chat, _, _, _ = self.assistant([
            tool_response("prepare_booking", booking_request()),
        ])
        self.prepare_and_confirm(chat)
        self.assert_occupied(1)
        reply = chat.reply("取消刚才保存的预订")
        self.assertEqual(reply.trace[-1]["status"], "cancelled")
        self.assertIn("已取消", reply.answer)
        self.assert_occupied(0)

    def test_new_session_cancellation_by_booking_id_releases_availability(self):
        owner, _, _, _ = self.assistant([
            tool_response("prepare_booking", booking_request()),
        ])
        saved = self.prepare_and_confirm(owner)
        booking_id = saved.trace[-1]["booking_id"]
        other, _, _, _ = self.assistant([])
        reply = other.reply(f"取消订单 {booking_id}")
        self.assertEqual(reply.trace[-1]["status"], "cancelled")
        self.assert_occupied(0)


if __name__ == "__main__":
    unittest.main()
