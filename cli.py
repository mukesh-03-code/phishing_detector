#!/usr/bin/env python3
"""
Standalone CLI Runner for Project 3: AI Phishing URL & Email Detector
Usage:
    python3 cli.py
    python3 cli.py --url "http://paypa1-login-verify.xyz/auth"
"""

import argparse
from ml_model import PhishingMLDetector


def main():
    parser = argparse.ArgumentParser(description="Project 3: AI Phishing URL & Email Detector")
    parser.add_argument("--url", default="", help="Single URL to analyze (runs demo suite if omitted)")
    args = parser.parse_args()

    detector = PhishingMLDetector()
    print("=" * 80)
    print(" [PHISHGUARD AI] Ensemble ML + Lexical Phishing Detection Engine")
    print(f" [MODEL METRICS] Accuracy: {detector.metrics['accuracy']}% | Precision: {detector.metrics['precision']}% | Recall: {detector.metrics['recall']}% | F1: {detector.metrics['f1_score']}%")
    print("=" * 80)

    test_urls = [args.url] if args.url else [
        "https://www.github.com/openai/whisper",
        "http://paypa1-account-security-verify.xyz/login?session=8821",
        "http://185.220.101.99/microsoft-office365/login.php",
        "https://hdfcbank-pan-kyc-update-urgent.buzz/netbanking",
    ]

    print("\n--- URL Analysis Results ---")
    for u in test_urls:
        res = detector.predict_url(u)
        print(f"\nURL      : {res['url']}")
        print(f"Verdict  : [{res['verdict']}] (Badge: {res['badge_color']}) | Risk Score: {res['risk_score']}% (ML Prob: {res['ml_probability']}%)")
        print(f"Brand    : {res['detected_brand_target']}")
        for r in res["reasons"]:
            print(f"  * {r}")

    if not args.url:
        print("\n--- Email NLP + Embedded URL Analysis Demo ---")
        email_res = detector.predict_email(
            subject="URGENT: Your PayPal Account Has Been Suspended - Action Required Within 24 Hours",
            sender="PayPal Security Center <alert@security-update-paypa1.xyz>",
            body=(
                "Dear Valued Customer,\n"
                "We detected an unauthorized login attempt on your account. Your account will be suspended within 24 hours "
                "unless you verify your identity and confirm your credit card immediately at:\n"
                "http://paypa1-account-security-verify.xyz/login?user=confirm\n"
                "Thank you."
            ),
        )
        print(f"\nSubject  : {email_res['subject']}")
        print(f"Sender   : {email_res['sender']}")
        print(f"Verdict  : [{email_res['verdict']}] | Combined Risk Score: {email_res['risk_score']}%")
        for r in email_res["reasons"]:
            print(f"  * {r}")


if __name__ == "__main__":
    main()
