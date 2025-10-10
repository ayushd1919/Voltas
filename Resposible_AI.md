# 🧭 Responsible AI Checklist  
**Project:** Voltas Availability Prediction  
**Version:** 1.0 | **Date:** 2025-10-10  
**Owners:** Data Science, Product, Security  

---

## 1️⃣ Purpose & Scope
- **Goal:** Predict if a product will be *In Stock* or *Out of Stock*.
- **Use Case:** Internal analytics for supply & operations planning.
- **Not for:** Credit, hiring, or other high-risk decisions.

---

## 2️⃣ Data Governance
- ✅ No personal identifiers (name, phone, address, email).
- ✅ Columns limited to product metadata (`price_inr`, `city`, `platform`, `energy_rating_stars`, etc.).
- ✅ Dataset source documented in `/app/artifacts/reference_sample.csv`.
- ✅ Artifacts tracked with DVC for reproducibility.

---

## 3️⃣ Fairness
### Sensitive Attributes
- Primary: `city`, `platform`
- Optional: `region`, `seller_type`

### Metrics
- **Demographic Parity Difference:** ≤ 0.10  
- **Equalized Odds Difference:** ≤ 0.10  
- **Action:**  
  - Green ≤ 0.10 → ✅ Fair  
  - Amber 0.10–0.20 → ⚠️ Review  
  - Red > 0.20 → ❌ Mitigation needed

### Mitigation
- **Pre-processing:** Reweight or stratify samples.
- **In-processing:** Fairlearn `ExponentiatedGradient` (DemographicParity).
- **Post-processing:** Threshold adjustments per group.

### Monitoring
- Run weekly fairness audit using Streamlit “⚖️ Fairness” tab.
- Auto-alert if fairness gap > 0.20.

---

## 4️⃣ Explainability
- Global explainability via **SHAP summary plot** (top feature drivers).
- Local explainability via **LIME or SHAP waterfall**.
- SHAP and metrics visualized in “🔎 SHAP” tab.
- Each explanation includes disclaimer:  
  > *Explanations are statistical approximations and not exact causal attributions.*

---

## 5️⃣ Privacy & Consent
- **PII:** None used or stored.
- **Consent:** Data derived from internal product listings and open data.
- **Anonymization:** No user-linked identifiers in dataset.
- **Access:** Controlled via organization GitHub/DVC permissions.
- **Secrets:** Stored in GitHub Actions → Encrypted (`GDRIVE_SA_JSON`, etc.).

---

## 6️⃣ Drift & Monitoring
- Drift tracked using **PSI (Population Stability Index)** and **KS test** in Streamlit “🌊 Drift” tab.
- Heuristic thresholds:
  - PSI ≥ 0.2 → High Drift  
  - PSI 0.1–0.2 → Medium Drift  
  - PSI < 0.1 → Stable

---

## 7️⃣ Safety & Misuse Prevention
- Predictions **for internal dashboards only** (non-automated).
- No direct integration with inventory control APIs.
- Human validation required for operational actions.
- Misuse prevention: rate limits, internal access only.

---

## 8️⃣ Responsible Deployment
- ✅ All tests and lint checks pass in CI/CD (`pytest`, `ruff`, `black`).
- ✅ No PII in artifacts.
- ✅ Fairness metrics within defined thresholds.
- ✅ Drift < 0.2 PSI.
- ✅ Explainability and visualization tested locally and in Streamlit Cloud.

---

## 9️⃣ Documentation & Transparency
- Model card and fairness summary saved under `/docs/`.
- Each deployment tagged (e.g., `v1.0.0`) with reproducible environment (`runtime.txt`, `requirements.txt`).
- Responsible_AI.md reviewed quarterly.

---

## 🔟 Contacts
| Role | Name | Contact |
|------|------|----------|
| Data Scientist | Ayush Duduskar | _ayushduduskar@gmail.com_ |
| Product Lead | Dheeraj Devdas | _dheerajdevdas@gmail.com_ |
| Security/Privacy | Aditya Gupta | _adityagupta@gmail.com_ |

---

### ✅ Final Review Checklist
| Check | Status |
|-------|---------|
| Model performance ≥ baseline | ☐ |
| Fairness thresholds met | ☐ |
| SHAP visualizations verified | ☐ |
| No personal data included | ☐ |
| Secrets secured in CI/CD | ☐ |
| Responsible_AI.md reviewed | ☐ |

---

> **Note:** This document should be updated whenever new datasets, models, or features are added.
