# PhishGuard AI — Phishing URL & Email Detector + Chrome Extension

A real-time anti-phishing system combining a **Scikit-Learn Voting Ensemble (`RandomForestClassifier` + `GradientBoostingClassifier`)** with **16 Lexical, Host, Homograph/Typosquatting, and Shannon Entropy Features**, an **Email NLP Social-Engineering Analyzer**, and a **Manifest V3 Chrome/Firefox Browser Extension**.

## Key Features
1. **16-Feature URL Extraction (`feature_extractor.py`):**
   - URL & Hostname length, dot/subdomain depth, hyphen count, `@` credential obfuscation, raw IPv4/Hex detection, HTTPS check, Shannon entropy, digit ratio, high-risk TLDs (`.xyz`, `.top`, `.tk`, `.buzz`, `.click`), homograph/typosquatting brand impersonation (`paypa1` -> `PayPal`), and social engineering keyword density.
2. **Email NLP Urgency & Spoofing Engine:**
   - Detects display-name vs. sender-domain mismatch (`PayPal Security <alert@evil.xyz>`), coercion language (`within 24 hours`, `account suspended`), and scans all embedded links.
3. **Manifest V3 Browser Extension (`chrome_extension/`):**
   - Load unpacked in `chrome://extensions` to get real-time Red/Yellow/Green trust badges and active warning banners.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run via CLI
```bash
python3 cli.py
```

### 3. Run API & Web Server (for Browser Extension)
```bash
python3 app.py
# Open http://localhost:8000
```

### 4. Run Unit Tests
```bash
python3 test_phishing.py
```
