from decimal import Decimal
import unittest

from app.pricing import PricingInputError, calculate_price


class CalculatePriceTests(unittest.TestCase):
    def test_medium_dog_five_days_has_no_discount(self):
        quote = calculate_price(pet_type="狗", dog_size="中型", days=5)

        self.assertEqual(quote.base_subtotal, Decimal("590.00"))
        self.assertEqual(quote.total, Decimal("590.00"))
        self.assertIsNone(quote.long_stay_discount_label)

    def test_holiday_price_is_raised_before_long_stay_discount(self):
        quote = calculate_price(
            pet_type="狗", dog_size="中型", days=10, holiday="春节"
        )

        self.assertEqual(quote.base_subtotal, Decimal("1180.00"))
        self.assertEqual(quote.holiday_adjustment, Decimal("354.00"))
        self.assertEqual(quote.long_stay_discount_amount, Decimal("153.40"))
        self.assertEqual(quote.total, Decimal("1380.60"))
        self.assertEqual(quote.long_stay_discount_label, "满7天，9折")

    def test_cat_deluxe_room_uses_fifteen_day_discount(self):
        quote = calculate_price(pet_type="猫", cat_room="豪华", days=15)

        self.assertEqual(quote.daily_rate, Decimal("98.00"))
        self.assertEqual(quote.total, Decimal("1249.50"))
        self.assertEqual(quote.long_stay_discount_label, "满15天，85折")

    def test_exotic_pet_thirty_days_uses_best_discount(self):
        quote = calculate_price(pet_type="异宠", days=30)

        self.assertEqual(quote.total, Decimal("960.00"))
        self.assertEqual(quote.long_stay_discount_label, "满30天及以上，8折")

    def test_dog_requires_a_size(self):
        with self.assertRaisesRegex(PricingInputError, "必须提供体型"):
            calculate_price(pet_type="狗", days=3)

    def test_invalid_days_are_rejected(self):
        with self.assertRaisesRegex(PricingInputError, "大于 0"):
            calculate_price(pet_type="猫", days=0)

    def test_unsupported_holiday_is_rejected(self):
        with self.assertRaisesRegex(PricingInputError, "暂不支持"):
            calculate_price(pet_type="猫", days=3, holiday="五一")


if __name__ == "__main__":
    unittest.main()
