"""
Project 3: AI/ML Phishing URL & Email Classifier
Combines a Scikit-Learn Ensemble (RandomForest + GradientBoosting) trained on
16 lexical/host/entropy URL features with explainable threat scoring.
"""

from typing import Dict
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

try:
    from .feature_extractor import extract_url_features, analyze_email_text
except ImportError:
    from feature_extractor import extract_url_features, analyze_email_text


LEGITIMATE_SEED_URLS = [
    "https://www.google.com/search?q=cybersecurity+projects",
    "https://github.com/torvalds/linux",
    "https://www.paypal.com/us/home",
    "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
    "https://www.apple.com/icloud/",
    "https://aws.amazon.com/console/",
    "https://www.netflix.com/browse",
    "https://www.linkedin.com/in/security-researcher",
    "https://stackoverflow.com/questions/tagged/python",
    "https://docs.python.org/3/library/sqlite3.html",
    "https://www.wikipedia.org/wiki/Intrusion_detection_system",
    "https://www.cloudflare.com/learning/ddos/what-is-a-ddos-attack/",
    "https://owasp.org/www-project-top-ten/",
    "https://nvd.nist.gov/vuln/search",
    "https://www.hdfcbank.com/personal/ways-to-bank",
    "https://onlinesbi.sbi/",
    "https://www.chase.com/personal/checking",
    "https://www.coinbase.com/price/bitcoin",
    "https://www.binance.com/en/markets",
    "https://ubuntu.com/download/server",
    "https://pypi.org/project/scikit-learn/",
    "https://mail.google.com/mail/u/0/#inbox",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP",
]

PHISHING_SEED_URLS = [
    "http://paypa1-secure-login-verify.xyz/webscr?cmd=_login-run&session=99812",
    "http://185.220.101.45/microsoft-office365-login/auth.php",
    "https://apple-icloud-unlock-account-support.top/signin",
    "http://secure-update-netflix-billing.tk/customer/confirm-card",
    "https://login.google.com.security-check-account.cf/ServiceLogin",
    "http://www.amazon-prime-refund-verify.gq/ap/signin?openid.mode=checkid",
    "http://hdfcbank-kyc-pan-update-urgent.buzz/netbanking/login.htm",
    "http://sbi-yono-account-blocked-verify.click/unlock-otp",
    "http://chase-online-banking-verify-identity.xyz/login/auth",
    "https://coinbase-wallet-seed-phrase-recover.top/connect",
    "http://binance-airdrop-claim-reward-login.pw/auth/verify",
    "http://facebook-security-notice-confirm-identity.ml/login.php",
    "http://192.168.55.102/admin/paypal/confirm_credentials.html",
    "http://https-www-paypal-com.account-verify-update.work/login",
    "https://microsoft-sharepoint-document-view.link/auth@evil-stealer.xyz/login",
    "http://github-security-oauth-verify-token.club/login/oauth",
    "http://instagram-verified-badge-apply-now.gq/accounts/login",
    "http://wellsfargo-alert-unauthorized-access.xyz/update//https://wellsfargo.com",
    "http://bit.ly/3xYz99FakeBankLogin",
    "http://0x7f000001/bank-login-verify-password/index.html",
    "http://citibank-account-suspended-urgent-update.loan/verify.php?user=admin",
    "https://meta-business-suite-policy-violation.win/appeal-account-verify",
    "http://dhl-express-parcel-customs-fee-payment.xyz/track/confirm-billing",
    "http://irs-tax-refund-portal-verify-ssn.top/claim-now",
]


class PhishingMLDetector:
    def __init__(self):
        rf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, max_depth=4, random_state=42)
        self.model = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb)],
            voting="soft",
        )
        self.metrics: Dict[str, float] = {}
        self._train_and_evaluate()

    def _build_training_dataset(self):
        X = []
        y = []
        rng = np.random.default_rng(42)

        for u in LEGITIMATE_SEED_URLS:
            vec = extract_url_features(u).to_numeric_list()
            X.append(vec)
            y.append(0)
            for _ in range(8):
                aug = list(vec)
                aug[0] = max(16.0, aug[0] + rng.integers(-6, 15))
                aug[1] = max(8.0, aug[1] + rng.integers(-2, 4))
                aug[8] = round(max(2.2, aug[8] + rng.uniform(-0.2, 0.2)), 3)
                X.append(aug)
                y.append(0)

        for u in PHISHING_SEED_URLS:
            vec = extract_url_features(u).to_numeric_list()
            X.append(vec)
            y.append(1)
            for _ in range(8):
                aug = list(vec)
                aug[0] = max(25.0, aug[0] + rng.integers(-5, 25))
                aug[3] = max(0.0, aug[3] + rng.integers(0, 2))
                aug[8] = round(max(3.2, aug[8] + rng.uniform(-0.15, 0.25)), 3)
                X.append(aug)
                y.append(1)

        return np.array(X, dtype=float), np.array(y, dtype=int)

    def _train_and_evaluate(self):
        X, y = self._build_training_dataset()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
            "precision": round(float(precision_score(y_test, y_pred)) * 100, 2),
            "recall": round(float(recall_score(y_test, y_pred)) * 100, 2),
            "f1_score": round(float(f1_score(y_test, y_pred)) * 100, 2),
            "training_samples": int(len(X)),
        }

    def predict_url(self, url: str) -> Dict:
        feat = extract_url_features(url)
        num_vec = np.array([feat.to_numeric_list()], dtype=float)
        ml_prob = float(self.model.predict_proba(num_vec)[0][1]) * 100.0

        heuristic_boost = 0.0
        if feat.brand_impersonation:
            heuristic_boost += 35.0
        if feat.has_ip_address:
            heuristic_boost += 30.0
        if feat.suspicious_tld:
            heuristic_boost += 20.0
        if feat.num_at_symbols > 0:
            heuristic_boost += 25.0
        if feat.phishing_keywords_count >= 2:
            heuristic_boost += 15.0

        risk_score = min(99.5, round(0.65 * ml_prob + min(35.0, heuristic_boost), 1))
        if not feat.risk_reasons and feat.uses_https:
            risk_score = min(risk_score, 12.0)

        if risk_score >= 65.0:
            verdict = "PHISHING"
            badge_color = "RED"
        elif risk_score >= 35.0:
            verdict = "SUSPICIOUS"
            badge_color = "YELLOW"
        else:
            verdict = "SAFE"
            badge_color = "GREEN"
            if not feat.risk_reasons:
                feat.risk_reasons.append("Valid HTTPS domain structure with no brand impersonation or anomalous lexical tokens.")

        return {
            "url": url,
            "verdict": verdict,
            "badge_color": badge_color,
            "risk_score": risk_score,
            "ml_probability": round(ml_prob, 1),
            "detected_brand_target": feat.detected_brand or "None",
            "reasons": feat.risk_reasons,
            "features": feat.to_dict(),
            "model_metrics": self.metrics,
        }

    def predict_email(self, subject: str, sender: str, body: str) -> Dict:
        email_info = analyze_email_text(subject, sender, body)
        url_results = [self.predict_url(u) for u in email_info["extracted_urls"]]

        max_url_risk = max((r["risk_score"] for r in url_results), default=0.0)
        combined_risk = min(99.5, round(max(email_info["nlp_risk_score"], max_url_risk * 0.9 + email_info["nlp_risk_score"] * 0.35), 1))

        if combined_risk >= 65.0:
            verdict = "PHISHING EMAIL"
        elif combined_risk >= 35.0:
            verdict = "SUSPICIOUS EMAIL"
        else:
            verdict = "LEGITIMATE EMAIL"

        all_reasons = list(email_info["matched_nlp_indicators"])
        for ur in url_results:
            if ur["verdict"] != "SAFE":
                all_reasons.append(f"Embedded URL '{ur['url']}' flagged as {ur['verdict']} (Risk: {ur['risk_score']}%).")

        if not all_reasons:
            all_reasons.append("No social engineering urgency phrases, sender spoofing, or malicious links detected.")

        return {
            "subject": subject,
            "sender": sender,
            "verdict": verdict,
            "risk_score": combined_risk,
            "nlp_risk_score": email_info["nlp_risk_score"],
            "reasons": all_reasons,
            "embedded_url_analyses": url_results,
        }
