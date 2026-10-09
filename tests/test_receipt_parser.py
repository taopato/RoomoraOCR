import unittest

from main import extract_items, ocr_result_score, try_extract_total


def receipt_lines(text: str):
    return [
        {
            "text": line,
            "left": 0,
            "top": index * 20,
            "width": 500,
            "height": 18,
        }
        for index, line in enumerate(text.strip().splitlines())
    ]


class ReceiptParserTests(unittest.TestCase):
    def test_prefers_result_with_more_confident_receipt_content(self):
        weak = [[[], "TOPLAM", 0.2]]
        strong = [[[], "ÖDENECEK TUTAR 1.411,00", 0.9]]

        self.assertGreater(ocr_result_score(strong), ocr_result_score(weak))

    def test_applies_repeated_discount_rows_to_matching_products(self):
        lines = receipt_lines(
            """
SLEEPY YUZTEM100LU %20.0 *119,00
BEYPAZARI 200 ML %1.0 *240,00
DANA KANGAL SUCUK %1.0 *419,00
URUN INDIRIMLERI:
BEYPAZARI 200 ML %1.00 *-6,00
BEYPAZARI 200 ML %1.00 *-6,00
BEYPAZARI 200 ML %1.00 *-6,00
DANA KANGAL SUCUK %1.00 *-119,50
SLEEPY YUZTEM100LU %20.0 *-41,50
MAL/HIZMET TOPLAM TUTARI *1.381,13
TOPKDV *29,87
ÖDENECEK TUTAR *1.411,00
"""
        )

        items = extract_items(lines)
        by_name = {item["name"].upper(): item for item in items}

        self.assertEqual(222.0, by_name["BEYPAZARI"]["line_total"])
        self.assertEqual(18.0, by_name["BEYPAZARI"]["discount_amount"])
        self.assertEqual(299.5, by_name["DANA KANGAL SUCUK"]["line_total"])
        self.assertEqual(77.5, by_name["SLEEPY YUZTEM100LU"]["line_total"])
        self.assertEqual(1411.0, try_extract_total(lines))

    def test_discount_section_treats_amount_as_discount_when_ocr_drops_minus(self):
        lines = receipt_lines(
            """
SUTAS TAM YAGLI SUT 1L *49,50
URUN INDIRIMLERI:
SUTAS TAM YAGLI SUT 1L *9,50
ÖDENECEK TUTAR *40,00
"""
        )

        items = extract_items(lines)

        self.assertEqual(40.0, items[0]["line_total"])
        self.assertEqual(9.5, items[0]["discount_amount"])

    def test_keeps_same_product_on_different_receipt_rows(self):
        lines = receipt_lines(
            """
KINDER JOY YUM. %1.0 *43,00
KINDER JOY YUM. %1.0 *43,00
"""
        )

        items = extract_items(lines)

        self.assertEqual(2, len(items))


if __name__ == "__main__":
    unittest.main()
