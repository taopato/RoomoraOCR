import unittest

from main import (
    extract_items,
    merge_lines,
    normalize_store_candidate,
    ocr_result_score,
    parse_amount,
    receipt_lines_look_upside_down,
    store_line_score,
    try_extract_total,
)


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
    def test_merges_product_and_distant_price_on_same_visual_row(self):
        entries = [
            {"text": "OTOPARK %20", "left": 100, "top": 200, "width": 180, "height": 32},
            {"text": "*70,00", "left": 720, "top": 196, "width": 120, "height": 34},
            {"text": "KDV", "left": 100, "top": 250, "width": 80, "height": 32},
            {"text": "*11,67", "left": 720, "top": 247, "width": 120, "height": 34},
        ]

        lines = merge_lines(entries)

        self.assertEqual(["OTOPARK %20 *70,00", "KDV *11,67"], [line["text"] for line in lines])
        item = extract_items(lines)[0]
        self.assertEqual(70.0, item["line_total"])
        self.assertEqual((720, 196, 120, 34), (
            item["box_left"], item["box_top"], item["box_width"], item["box_height"]
        ))

    def test_accepts_ocr_space_after_decimal_separator(self):
        items = extract_items(receipt_lines("CANTA %8 *350, 00"))

        self.assertEqual(350.0, items[0]["line_total"])

    def test_parses_turkish_and_international_thousands(self):
        self.assertEqual(1355.0, parse_amount("1.355,00"))
        self.assertEqual(1355.0, parse_amount("1,355.00"))

    def test_does_not_parse_three_decimal_weight_as_money(self):
        items = extract_items(receipt_lines("ELMA 0,754 kg x 19,95 %1 *15,04"))

        self.assertEqual(15.04, items[0]["line_total"])

    def test_ignores_tax_rate_when_extracting_item_price(self):
        items = extract_items(receipt_lines("HOBBY LIFE KRO %18,00 *99,00"))

        self.assertEqual(99.0, items[0]["line_total"])

    def test_does_not_treat_top_summary_as_product(self):
        items = extract_items(receipt_lines("OTOPARK %20 *70,00\nTOP *70,00"))

        self.assertEqual(["OTOPARK"], [item["name"] for item in items])

    def test_does_not_treat_poset_as_pos_terminal(self):
        items = extract_items(receipt_lines("ALISVERIS POSETI %20 *2,00"))

        self.assertEqual(2.0, items[0]["line_total"])

    def test_prefers_merchant_over_document_heading(self):
        self.assertGreater(store_line_score("A101 YENI MAGAZACILIK A.S."), store_line_score("e-Arsiv Fatura"))

    def test_splits_merchant_from_merged_address(self):
        text = "58 BESLER GIDA AKR INS TAAH TIC LTD STI ISTIKLAL MAH DORTEYLUL CD. NO:22/A"

        self.assertEqual("58 BESLER GIDA AKR INS TAAH TIC LTD STI ISTIKLAL", normalize_store_candidate(text))

    def test_detects_upside_down_receipt_order(self):
        lines = receipt_lines(
            """
TOPLAM *81,61
KREDI KARTI *81,61
URUN A *20,00
URUN B *61,61
FIS NO 12
SAAT 14:18
22.05.2023
58 BESLER MARKET
"""
        )

        self.assertTrue(receipt_lines_look_upside_down(lines))

    def test_total_can_follow_a_detached_label(self):
        lines = receipt_lines(
            """
PILIC KANAT *97,90
TOPLAM
*3,44
*347,54
"""
        )

        self.assertEqual(347.54, try_extract_total(lines))

    def test_tax_total_is_not_payable_total(self):
        lines = receipt_lines(
            """
KDV TOPLAM *0,79
KREDI *80,00
"""
        )

        self.assertEqual(80.0, try_extract_total(lines))

    def test_merged_tax_and_total_line_uses_larger_amount(self):
        lines = receipt_lines(
            """
TOPKDV *1252,98 *93,05
TOPLAM
NAKIT *1300,00 *47,02
"""
        )

        self.assertEqual(1252.98, try_extract_total(lines))

    def test_prefers_total_amount_over_total_discount(self):
        lines = receipt_lines(
            """
Toplam Iskonto 1 140.00 TL
Toplam Tutar 381,98 TL
"""
        )

        self.assertEqual(381.98, try_extract_total(lines))

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

    def test_fuzzy_matches_a_discount_name_corrupted_differently_by_ocr(self):
        lines = receipt_lines(
            """
SLEEPYY\ufffdZTEM10OL\ufffd %20.0 *119,00
URUN INDIRIMLERI:
SLEEPYYUZTEM1OOL\ufffd %20.0 *-41,50
ODENECEK TUTAR *77,50
"""
        )

        items = extract_items(lines)

        self.assertEqual(77.5, items[0]["line_total"])
        self.assertEqual(41.5, items[0]["discount_amount"])

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
