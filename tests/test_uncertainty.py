import unittest

from fudu.uncertainty import box_uncertainty, classification_entropy, image_defect_uncertainty, localization_entropy


class UncertaintyTest(unittest.TestCase):
    def test_classification_entropy_bounds(self):
        self.assertAlmostEqual(classification_entropy([1.0, 0.0, 0.0]), 0.0)
        self.assertAlmostEqual(classification_entropy([1.0, 1.0, 1.0]), 1.0)

    def test_localization_entropy_bounds(self):
        self.assertAlmostEqual(localization_entropy([[1.0, 0.0], [1.0, 0.0]]), 0.0)
        self.assertAlmostEqual(localization_entropy([[0.5, 0.5], [0.5, 0.5]]), 1.0)

    def test_box_uncertainty_uses_confidence(self):
        value = box_uncertainty(class_probs=[0.9, 0.1], loc_entropy_norm=0.0, w_cls=0.0, w_loc=0.0)
        self.assertAlmostEqual(value, (1.0 - 0.9) / 3.0)

    def test_image_defect_uncertainty_empty_boxes(self):
        self.assertEqual(image_defect_uncertainty([]), 0.0)


if __name__ == "__main__":
    unittest.main()

