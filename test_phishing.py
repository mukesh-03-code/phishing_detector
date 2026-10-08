#!/usr/bin/env python3
import unittest
from ml_model import PhishingMLDetector


class TestPhishingDetector(unittest.TestCase):
    def test_predictions(self):
        detector = PhishingMLDetector()
        safe = detector.predict_url("https://www.github.com/torvalds/linux")
        self.assertEqual(safe["verdict"], "SAFE")
        phish = detector.predict_url("http://paypa1-account-security-verify.xyz/login")
        self.assertEqual(phish["verdict"], "PHISHING")


if __name__ == "__main__":
    unittest.main()
