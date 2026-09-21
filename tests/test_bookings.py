from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import unittest

from app.bookings import BookingInputError, BookingStore, ROOM_LABELS


class BookingStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "bookings.sqlite3"
        self.store = BookingStore(self.db_path)

    def test_new_store_has_no_occupied_positions(self):
        availability = self.store.check_availability("cat_standard", "2026-10-01", "2026-10-04")
        self.assertEqual(availability["status"], "available")
        self.assertTrue(availability["available"])
        self.assertEqual(availability["capacity"], 2)
        self.assertEqual(len(availability["days"]), 4)
        self.assertTrue(all(day["occupied"] == 0 and day["remaining"] == 2 for day in availability["days"]))

    def test_existing_and_reopened_instances_read_committed_orders(self):
        customer_b = BookingStore(self.db_path)
        booked = self.store.create_booking("cat_standard", "2026-10-01", "2026-10-04", "request-a")
        self.assertEqual(booked["status"], "booked")
        for other in (customer_b, BookingStore(self.db_path)):
            availability = other.check_availability("cat_standard", "2026-10-01", "2026-10-04")
            self.assertTrue(all(day["occupied"] == 1 and day["remaining"] == 1 for day in availability["days"]))
            self.assertTrue(availability["available"])

    def test_last_date_is_occupied_and_next_day_is_free(self):
        self.store.create_booking("cat_deluxe", "2026-10-01", "2026-10-04", "request-a")
        availability = self.store.check_availability("cat_deluxe", "2026-10-04", "2026-10-05")
        self.assertEqual(availability["days"], [
            {"date": "2026-10-04", "occupied": 1, "remaining": 0},
            {"date": "2026-10-05", "occupied": 0, "remaining": 1},
        ])
        self.assertEqual(availability["status"], "partial")
        self.assertFalse(availability["available"])

    def test_partial_full_range_cannot_be_booked(self):
        self.store.create_booking("cat_deluxe", "2026-10-02", "2026-10-03", "request-a")
        failed = self.store.create_booking("cat_deluxe", "2026-10-01", "2026-10-04", "request-b")
        self.assertEqual(failed["status"], "unavailable")
        self.assertEqual(failed["availability"]["status"], "partial")
        self.assertEqual(self.store.check_availability("cat_deluxe", "2026-10-01", "2026-10-01")["days"][0]["occupied"], 0)

    def test_rooms_have_independent_capacity_and_no_other_order_details(self):
        self.store.create_booking("cat_deluxe", "2026-10-01", "2026-10-04", "request-a")
        for room_type in ROOM_LABELS:
            result = self.store.check_availability(room_type, "2026-10-01", "2026-10-04")
            self.assertEqual(result["status"], "full" if room_type == "cat_deluxe" else "available")
            self.assertNotIn("booking_id", result)
            self.assertNotIn("request_id", result)
            self.assertTrue(all(set(day) == {"date", "occupied", "remaining"} for day in result["days"]))

    def test_adjacent_nonoverlapping_ranges_can_be_booked(self):
        self.store.create_booking("cat_deluxe", "2026-10-01", "2026-10-04", "request-a")
        result = self.store.create_booking("cat_deluxe", "2026-10-05", "2026-10-07", "request-b")
        self.assertEqual(result["status"], "booked")
        self.assertTrue(all(day["occupied"] == 1 for day in self.store.check_availability("cat_deluxe", "2026-10-01", "2026-10-07")["days"]))

    def test_repeated_submission_is_idempotent_even_when_full(self):
        first = self.store.create_booking("cat_deluxe", "2026-10-01", "2026-10-04", "request-a")
        again = BookingStore(self.db_path).create_booking("cat_deluxe", "2026-10-01", "2026-10-04", "request-a")
        self.assertEqual(again["status"], "already_booked")
        self.assertEqual(first["booking_id"], again["booking_id"])
        self.assertTrue(all(day["occupied"] == 1 for day in self.store.check_availability("cat_deluxe", "2026-10-01", "2026-10-04")["days"]))

    def test_reused_request_id_with_different_parameters_is_rejected(self):
        self.store.create_booking("cat_standard", "2026-10-01", "2026-10-04", "request-a")
        for room, first, last in (
            ("dog_medium", "2026-10-01", "2026-10-04"),
            ("cat_standard", "2026-10-02", "2026-10-04"),
            ("cat_standard", "2026-10-01", "2026-10-05"),
        ):
            with self.subTest(room=room, first=first, last=last):
                with self.assertRaises(BookingInputError):
                    self.store.create_booking(room, first, last, "request-a")
        self.assertEqual(self.store.create_booking("dog_medium", "2026-10-01", "2026-10-04", "request-b")["status"], "booked")

    def test_invalid_dates_are_rejected_by_query_and_create(self):
        for first, last in (
            ("", "2026-10-04"), (None, "2026-10-04"),
            ("2026-2-01", "2026-10-04"), ("20260201", "2026-10-04"),
            ("2026-02-29", "2026-03-01"), ("2026-10-05", "2026-10-04"),
            ("2026-10-01", "2026-13-01"), ("2026-10-01", "2026-10-01T12:00:00"),
            ("0000-01-01", "2026-10-04"), ("2026-01-01", "2027-01-02"),
        ):
            with self.subTest(first=first, last=last):
                with self.assertRaises(BookingInputError):
                    self.store.check_availability("cat_standard", first, last)
                with self.assertRaises(BookingInputError):
                    self.store.create_booking("cat_standard", first, last, "bad-date")

    def test_same_day_and_calendar_edges_work(self):
        for day in ("2028-02-29", "2026-12-31", "9999-12-31"):
            with self.subTest(day=day):
                availability = self.store.check_availability("cat_standard", day, day)
                self.assertEqual(len(availability["days"]), 1)
                self.assertEqual(availability["days"][0]["date"], day)

    def test_invalid_room_or_request_id_is_rejected(self):
        for room in (None, "", "未知", [], 1):
            with self.subTest(room=room):
                with self.assertRaises(BookingInputError):
                    self.store.check_availability(room, "2026-10-01", "2026-10-04")
        for request_id in (None, "", "  ", [], "x" * 129):
            with self.subTest(request_id=request_id):
                with self.assertRaises(BookingInputError):
                    self.store.create_booking("cat_standard", "2026-10-01", "2026-10-04", request_id)

    def test_concurrent_customers_cannot_sell_the_last_position_twice(self):
        barrier = threading.Barrier(2)
        stores = [BookingStore(self.db_path), BookingStore(self.db_path)]

        def reserve(index):
            barrier.wait(timeout=10)
            return stores[index].create_booking("cat_deluxe", "2026-10-01", "2026-10-04", f"concurrent-{index}")

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(reserve, range(2)))
        self.assertEqual(sorted(result["status"] for result in results), ["booked", "unavailable"])
        self.assertTrue(all(day["occupied"] == 1 for day in self.store.check_availability("cat_deluxe", "2026-10-01", "2026-10-04")["days"]))


if __name__ == "__main__":
    unittest.main()
