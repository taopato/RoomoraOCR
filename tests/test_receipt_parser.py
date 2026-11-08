import unittest
import numpy as np

from main import (
    discard_implausible_items,
    extract_items,
    merge_lines,
    normalize_store_candidate,
    normalize_product_candidate,
    ocr_result_score,
    parse_amount,
    receipt_lines_look_upside_down,
    recover_missing_priced_items,
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
    def test_does_not_recover_existing_discounted_product_as_new_item(self):
        image = np.zeros((1000, 1500, 3), dtype=np.uint8)
        lines = [{"text": "SLEEPY %20 *119,00", "left": 700, "top": 500,
                  "width": 500, "height": 30}]
        existing = [{"name": "SLEEPY", "line_total": 77.5, "box_top": 500}]
        calls = []

        result = recover_missing_priced_items(image, lambda crop: calls.append(crop) or [],
                                              lambda value: value, lines, existing, 196.5)

        self.assertEqual(existing, result)
        self.assertEqual([], calls)

    def test_recovers_faint_product_name_from_short_crop(self):
        image = np.zeros((1000, 1500, 3), dtype=np.uint8)
        lines = [{"text": "8695855505897 %20 *125,00", "left": 760, "top": 500,
                  "width": 550, "height": 30}]
        existing = [{"name": "SOFIA", "line_total": 250.0, "box_top": 560}]
        cropped_ocr = [
            [[[760, 100], [1050, 100], [1050, 130], [760, 130]], "ONLY EVO 12'LI KAG", 0.99],
            [[[1220, 100], [1320, 100], [1320, 130], [1220, 130]], "*125,00", 0.99],
        ]

        result = recover_missing_priced_items(image, lambda _: cropped_ocr, lambda value: value,
                                              lines, existing, 375.0)

        self.assertEqual(2, len(result))
        self.assertEqual("ONLY EVO 12'LI KAG", result[0]["name"])
        self.assertEqual(500, result[0]["box_top"])

    def test_removes_barcodes_embedded_in_product_name(self):
        self.assertEqual("ONLY EVO 12'LI KAG", normalize_product_candidate("ONLY EVO 12'LI KAG 8695855505897"))

    def test_hanci_market_items_match_printed_total_without_tax_rows(self):
        lines = receipt_lines("""
OZHANCI MARKET ISL.BAK.SAN.TIC.LTD.STI
TARIH : 07.11.2026 SAAT : 18:48
0,414 KGx 39,95
KURU SOGAN %01 *16,54
0,692 KGx 174,95
SALKIM DOMATES %01 *121,07
0,776 KGx 39,95
KURU SOGAN %01 *31,00
ONLY EVO 12'LI KAG. %20 *125,00
SOFIA 32'LI TUVALET %20 *250,00
CALVE TATLI KECAP %01 *185,00
THORLIGHT 20 LED BEY %20 *110,00
ROLL-UP 8M SILIKONLU %20 *65,00
ROLL-UP 8M SILIKONLU %20 *65,00
FUJIKA 40 W LED AMPU %20 *150,00
BILLUR TUZ IYOTLU TU %01 *30,00
ARKO TIRAS KOP. 200 %20 *115,00
ZAMBAK 6'LI PLASTIK %20 *75,00
ZAMBAK 6'LI PLASTIK %20 *75,00
FLOREX 10'LU COP T. %20 *45,00
FLOREX 10'LU COP T. %20 *45,00
FLOREX 10'LU COP T. %20 *45,00
22010UB0443AD OLCU C %20 *150,00
MASTERINI 200 ML BIT %01 *25,00
MASTERINI 200 ML BIT %01 *25,00
KOSKA 400 GR PRATIK %01 *120,00
0,445 KGx 225,00
BAGCI BIBERLI ZEYT. %01 *100,13
THORLIGHT 12 LED BEY %20 *50,00
ERK YUMURTA 15'LI %01 *75,00
SIGNAL 75 ML DIS M. %10 *25,00
ECE 70'LI BUYUK BUZD %20 *30,00
BILLUR TUZ 750 GR %01 *25,00
KNORR 60 GR KARABIBE %01 *90,00
PINAR SERVIS SOSIS H %01 *195,00
SIGNAL 1+1 DIS F. PE %10 *160,00
ATC PIL 4'LU INCE PI %20 *45,00
ATC PIL 4'LU INCE PI %20 *45,00
TOPKDV *274,60
TOPLAM *2708,74
BANKA KREDI KARTI *2708,74
KDV KDV TUTARI KDV'LI TOPLAM
%01 *10,28 *1038,74
%10 *16,82 *185,00
%20 *247,50 *1485,00
""")

        items = extract_items(lines)

        self.assertEqual(32, len(items))
        self.assertEqual(2708.74, round(sum(item["line_total"] for item in items), 2))
        self.assertEqual(3, sum("FLOREX" in item["name"].upper() for item in items))
        self.assertEqual(2708.74, try_extract_total(lines))

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

    def test_a101_subtotal_is_not_a_product(self):
        lines = receipt_lines(
            """ALISVERIS POSETI %20 *1,00
AKIM KORUMALI 3 PRIZ %20 *299,00
K. PILIC BONFILE %01 *153,44
PIR PIL YERL 1KG OVA %01 *57,50
PIZZAKING600GSF %01 *199,00
BALIK6X75GDARDAN %01 *645,00
SLEEPYYUZTEM100LU %20 *119,00
ARA TOP. *1.473,94
URUN INDIRIMLERI:
SLEEPYYUZTEM100LU *41,50-D
BALIK6X75GDARDAN *320,00-D
TOPLAM *1112,44"""
        )
        items = extract_items(lines)
        self.assertEqual(1112.44, round(sum(item["line_total"] for item in items), 2))
        self.assertFalse(any("ARA TOP" in item["name"] for item in items))

    def test_migros_correction_and_general_discount(self):
        lines = receipt_lines(
            """DORITOS MEGA BOY DORITOS NACHO MEGA *91,45
*91,45
WINSTON XSENCE BLACK *130,00
PRINGLES SOUR CREAM *149,90
**HATA DUZELT**
PRINGLES SOUR CREAM x-149,90
UNTAD LAVAS 8LI *139,50
MASTER POTATO SOG *125,00
INDIRIM *-35,50
INDIRIMLER:
UNTAD LAVAS 8LI *-60,00
TOPKDV TOPLAM *481,90 *3,48"""
        )
        items = extract_items(lines)
        self.assertEqual(481.9, round(sum(item["line_total"] for item in items), 2))
        self.assertEqual(481.9, try_extract_total(lines))
        self.assertFalse(any("PRINGLES" in item["name"] for item in items))
        self.assertEqual(2, sum("DORITOS" in item["name"] for item in items))

    def test_payment_terminal_lines_are_not_products(self):
        lines = receipt_lines(
            """CANTA %8 *350,00
TOPLAM *350,00
DENIZBANK *350,00
ISLEM TUTARI 350,00 TL
TOPLM 350,00"""
        )
        self.assertEqual(["CANTA"], [item["name"] for item in extract_items(lines)])

    def test_bank_tax_and_unit_price_lines_are_not_products(self):
        lines = receipt_lines(
            """ULKER CIKOLATA *183,03
TL./KG *199,00
KDU *20,59
YAPIKREDI *183,03
Finansbank Pesin (1) KREDt KARTI *183,03
alt ECore Cashless *183,03"""
        )
        self.assertEqual(["ULKER CIKOLATA"], [item["name"] for item in extract_items(lines)])

    def test_cancelled_product_is_removed(self):
        lines = receipt_lines(
            """OKYANUS KOSELIK *185,00
** IPTAL **
OKYANUS KOSELIK *-185,00
HAVLULUK *99,00
TOPLAM *99,00"""
        )
        self.assertEqual(["HAVLULUK"], [item["name"] for item in extract_items(lines)])

    def test_invoice_quantity_columns_are_not_products(self):
        lines = receipt_lines("1 Ad 88380,01\n1 Ade 759,99\nTOPLAM TUTAR 381,98 TL")
        self.assertEqual([], extract_items(lines))

    def test_implausible_invoice_amount_requires_manual_review(self):
        items = [{"name": "ODA KOKUSU", "line_total": 88380.01}]
        self.assertEqual([], discard_implausible_items(items, 381.98))
        self.assertEqual(items, discard_implausible_items(items, 90000))


if __name__ == "__main__":
    unittest.main()
