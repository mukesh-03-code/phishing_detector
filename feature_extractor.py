"""
Project 3: Phishing URL & Email Feature Extractor
Extracts 16 lexical, structural, Shannon entropy, brand-typosquatting, and TLD features
from URLs, plus NLP social-engineering urgency indicators from Email bodies.
"""

import math
import re
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple


SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "pw", "cc", "club",
    "work", "buzz", "click", "link", "zip", "mov", "kim", "loan", "win",
}

PROTECTED_BRANDS = {
    "paypal": ["paypal.com", "paypal.me"],
    "microsoft": ["microsoft.com", "live.com", "office.com", "outlook.com"],
    "apple": ["apple.com", "icloud.com"],
    "google": ["google.com", "gmail.com", "youtube.com", "googleapis.com"],
    "amazon": ["amazon.com", "amazon.in", "aws.amazon.com"],
    "netflix": ["netflix.com"],
    "facebook": ["facebook.com", "fb.com", "meta.com"],
    "instagram": ["instagram.com"],
    "github": ["github.com", "githubusercontent.com"],
    "linkedin": ["linkedin.com"],
    "coinbase": ["coinbase.com"],
    "binance": ["binance.com"],
    "chase": ["chase.com"],
    "sbi": ["onlinesbi.sbi", "sbi.co.in"],
    "hdfc": ["hdfcbank.com"],
}

HOMOGRAPH_SUBSTITUTIONS = {
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "@": "a",
    "rn": "m",
    "vv": "w",
}

PHISHING_URL_KEYWORDS = [
    "login", "signin", "verify", "verification", "update", "secure", "account",
    "banking", "confirm", "suspended", "unlock", "wallet", "password", "credential",
    "webscr", "auth", "recover", "billing", "invoice", "support-ticket",
]

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rb.gy", "cutt.ly"
}

EMAIL_URGENCY_PATTERNS = [
    (r"within\s+\d+\s+(hours?|minutes?)", "Artificial time constraint ('within X hours')"),
    (r"(account|access|profile)\s+(has\s+been\s+|will\s+be\s+)?(suspended|locked|terminated|restricted|deactivated)", "Account suspension threat"),
    (r"(immediate|urgent)\s+(action|attention|verification)\s+required", "High-urgency coercion phrase"),
    (r"unauthorized\s+(login|sign-in|transaction|access)\s+attempt", "Security scare tactic ('unauthorized login attempt')"),
    (r"verify\s+your\s+(identity|account|payment|wallet|credentials)", "Direct credential harvesting call-to-action"),
    (r"(confirm|update)\s+your\s+(billing|credit\s+card|bank|ssn|tax|password)", "Financial / sensitive data request"),
    (r"(seed\s+phrase|private\s+key|recovery\s+phrase|otp|cvv\s+code)", "High-risk secret / crypto / OTP solicitation"),
    (r"dear\s+(valued\s+)?(customer|user|client|member|account\s+holder)", "Generic impersonal salutation ('Dear Customer')"),
]


def compute_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = defaultdict(int)
    for ch in text:
        counts[ch] += 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def normalize_homographs(domain_text: str) -> str:
    normalized = domain_text.lower()
    for fake, real in HOMOGRAPH_SUBSTITUTIONS.items():
        normalized = normalized.replace(fake, real)
    return normalized


def extract_registered_domain(hostname: str) -> str:
    parts = [p for p in hostname.lower().split(".") if p]
    if len(parts) >= 2:
        if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "net", "gov", "ac"):
            return ".".join(parts[-3:])
        return ".".join(parts[-2:])
    return hostname.lower()


@dataclass
class URLFeatureVector:
    url: str
    url_length: int
    hostname_length: int
    num_dots: int
    num_hyphens: int
    num_at_symbols: int
    has_ip_address: int
    uses_https: int
    num_subdomains: int
    shannon_entropy: float
    digit_ratio: float
    suspicious_tld: int
    brand_impersonation: int
    phishing_keywords_count: int
    has_double_slash_redirect: int
    is_url_shortener: int
    query_length: int
    detected_brand: str
    risk_reasons: List[str]

    def to_numeric_list(self) -> List[float]:
        return [
            float(self.url_length),
            float(self.hostname_length),
            float(self.num_dots),
            float(self.num_hyphens),
            float(self.num_at_symbols),
            float(self.has_ip_address),
            float(self.uses_https),
            float(self.num_subdomains),
            float(self.shannon_entropy),
            float(self.digit_ratio),
            float(self.suspicious_tld),
            float(self.brand_impersonation),
            float(self.phishing_keywords_count),
            float(self.has_double_slash_redirect),
            float(self.is_url_shortener),
            float(self.query_length),
        ]

    def to_dict(self) -> Dict:
        return asdict(self)


def extract_url_features(raw_url: str) -> URLFeatureVector:
    url = raw_url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url

    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    hostname = parsed.hostname.lower() if parsed.hostname else netloc.split(":")[0]
    path_and_query = (parsed.path or "") + ("?" + parsed.query if parsed.query else "")

    url_length = len(url)
    hostname_length = len(hostname)
    num_dots = hostname.count(".")
    num_hyphens = hostname.count("-")
    num_at_symbols = url.count("@")

    # IPv4 or hex IP check
    has_ip = 1 if re.match(r"^(\d{1,3}\.){3}\d{1,3}$", hostname) or re.match(r"^0x[0-9a-f]+", hostname) else 0
    uses_https = 1 if scheme == "https" else 0

    host_parts = [p for p in hostname.split(".") if p]
    num_subdomains = max(0, len(host_parts) - 2) if not has_ip else 0
    tld = host_parts[-1] if host_parts and not has_ip else ""

    entropy = round(compute_entropy(hostname + parsed.path), 3)
    digits = sum(1 for c in hostname if c.isdigit())
    digit_ratio = round(digits / max(len(hostname), 1), 3)

    suspicious_tld = 1 if tld in SUSPICIOUS_TLDS else 0
    is_shortener = 1 if hostname in URL_SHORTENERS else 0

    # Check brand impersonation / homograph / typosquatting
    reg_domain = extract_registered_domain(hostname)
    normalized_host = normalize_homographs(hostname)
    full_normalized = normalize_homographs(hostname + parsed.path)

    brand_impersonation = 0
    detected_brand = ""
    for brand, official_domains in PROTECTED_BRANDS.items():
        if reg_domain in official_domains or hostname in official_domains:
            continue
        if brand in hostname or brand in normalized_host or brand in full_normalized:
            brand_impersonation = 1
            detected_brand = brand.capitalize()
            break

    url_lower = url.lower()
    kw_matches = [kw for kw in PHISHING_URL_KEYWORDS if kw in url_lower]
    phishing_keywords_count = len(kw_matches)

    has_double_slash = 1 if "//" in path_and_query else 0
    query_length = len(parsed.query)

    reasons: List[str] = []
    if brand_impersonation:
        reasons.append(f"Brand impersonation / typosquatting detected for '{detected_brand}' on unauthorized domain '{reg_domain}'.")
    if has_ip:
        reasons.append(f"URL uses raw IP address ({hostname}) instead of a registered domain name.")
    if num_at_symbols > 0:
        reasons.append("URL contains '@' symbol used to obscure the true destination host.")
    if suspicious_tld:
        reasons.append(f"Domain uses high-risk top-level domain (.{tld}) frequently abused in phishing campaigns.")
    if num_hyphens >= 2:
        reasons.append(f"Hostname contains {num_hyphens} hyphens ('-'), common in deceptive combo-squatting domains.")
    if num_subdomains >= 3:
        reasons.append(f"Excessive subdomain depth ({num_subdomains} subdomains) used to mimic legitimate URLs.")
    if not uses_https:
        reasons.append("Connection uses unencrypted HTTP protocol.")
    if phishing_keywords_count >= 2:
        reasons.append(f"URL contains {phishing_keywords_count} sensitive social-engineering keywords ({', '.join(kw_matches[:4])}).")
    if entropy > 4.35:
        reasons.append(f"High character randomness (Shannon entropy = {entropy} bits/char), typical of auto-generated phishing URLs.")
    if has_double_slash:
        reasons.append("Embedded '//' redirect sequence detected inside URL path.")
    if is_shortener:
        reasons.append(f"URL uses link shortener ({hostname}) which hides the final destination.")

    return URLFeatureVector(
        url=raw_url,
        url_length=url_length,
        hostname_length=hostname_length,
        num_dots=num_dots,
        num_hyphens=num_hyphens,
        num_at_symbols=num_at_symbols,
        has_ip_address=has_ip,
        uses_https=uses_https,
        num_subdomains=num_subdomains,
        shannon_entropy=entropy,
        digit_ratio=digit_ratio,
        suspicious_tld=suspicious_tld,
        brand_impersonation=brand_impersonation,
        phishing_keywords_count=phishing_keywords_count,
        has_double_slash_redirect=has_double_slash,
        is_url_shortener=is_shortener,
        query_length=query_length,
        detected_brand=detected_brand,
        risk_reasons=reasons,
    )


def analyze_email_text(subject: str, sender: str, body: str) -> Dict:
    """Analyzes email subject, sender header, and body text for phishing indicators and embedded malicious URLs."""
    full_text = f"{subject}\n{body}"
    matched_indicators: List[str] = []
    urgency_score = 0

    for pattern, label in EMAIL_URGENCY_PATTERNS:
        if re.search(pattern, full_text, re.IGNORECASE):
            matched_indicators.append(label)
            urgency_score += 22

    # Check sender spoofing (e.g. "PayPal Support <admin@secure-alert-verify.xyz>")
    sender_lower = sender.lower()
    if "<" in sender_lower and ">" in sender_lower:
        display_part = sender_lower.split("<")[0]
        addr_part = sender_lower.split("<")[1].split(">")[0]
        addr_domain = addr_part.split("@")[-1] if "@" in addr_part else ""
        for brand, official_list in PROTECTED_BRANDS.items():
            if brand in display_part and not any(addr_domain.endswith(off) for off in official_list):
                matched_indicators.append(
                    f"Sender Display Name Spoofing: claims '{brand.capitalize()}' but sent from '{addr_domain}'."
                )
                urgency_score += 35

    # Extract embedded URLs in email body
    extracted_urls = re.findall(r"https?://[^\s<>\"']+", body)
    return {
        "sender": sender,
        "subject": subject,
        "matched_nlp_indicators": matched_indicators,
        "nlp_risk_score": min(100, urgency_score),
        "extracted_urls": extracted_urls,
    }
