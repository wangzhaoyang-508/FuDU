import unittest

from fudu.fuzzy import FuzzySampler, trapezoid


class FuzzyTest(unittest.TestCase):
    def test_trapezoid_boundary(self):
        self.assertEqual(trapezoid(0.0, 0.0, 0.0, 0.1, 0.2), 1.0)
        self.assertAlmostEqual(trapezoid(0.15, 0.0, 0.0, 0.1, 0.2), 0.5)

    def test_default_rules(self):
        sampler = FuzzySampler()
        self.assertEqual(sampler.evaluate(0.05, 0.05).action, "DNS")
        self.assertEqual(sampler.evaluate(0.45, 0.15).action, "HS")
        self.assertEqual(sampler.evaluate(0.79, 0.20).action, "MS")

    def test_preset(self):
        sampler = FuzzySampler.from_preset("eles")
        self.assertEqual(sampler.evaluate(0.70, 0.1).action, "MS")


if __name__ == "__main__":
    unittest.main()
