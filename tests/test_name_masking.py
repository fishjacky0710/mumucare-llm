import unittest
from unittest.mock import patch

import ifsp_isp_ai as core


class NameMaskingTests(unittest.TestCase):
    def test_preserves_ordinary_words(self):
        text = "維持站立安全與站立活動安全。"
        self.assertEqual(core.mask_known_names(text, ["陳立安"]), text)

    def test_masks_full_name_inside_sentence(self):
        self.assertEqual(core.mask_known_names("陳立安參與活動", ["陳立安"]), "**參與活動")

    def test_masks_delimited_given_name(self):
        self.assertEqual(core.mask_known_names("立安，請休息。", ["陳立安"]), "**，請休息。")

    def test_masks_name_field(self):
        self.assertEqual(core.mask_known_names("姓名：立安參與活動", ["陳立安"]), "姓名：**參與活動")

    def test_does_not_mask_single_character_tail(self):
        self.assertEqual(core.mask_known_names("安全與平安", ["王安"]), "安全與平安")

    def test_excel_entry_point_uses_shared_rules(self):
        with patch.object(core, "get_full_names", return_value=("陳立安",)):
            self.assertEqual(core.replace_name_with_stars("陳立安維持站立安全"), "**維持站立安全")
