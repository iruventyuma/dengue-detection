# 🦟 DengueDx — Hematological Dengue Detection System

A machine learning web application that predicts dengue fever from **Complete Blood Count (CBC)** test values using a stacked ensemble model. Built with Python, Scikit-learn, and Streamlit.

---

## 🩺 Live Demo

> Run locally with `streamlit run app.py`

---

## 🧠 Model Architecture

This project uses a **dual-path stacked ensemble** approach:

```
Raw CBC Features ──► ExtraTreesClassifier ──► Probabilities ─┐
                                                               ├──► MLPClassifier (Meta Learner) ──► Prediction
Ratio Features ───► GradientBoostingClassifier ─► Probabilities ─┘
```

| Layer | Model | Purpose |
|-------|-------|---------|
| Path 1 | ExtraTreesClassifier (500 trees) | Raw blood parameters |
| Path 2 | GradientBoostingClassifier (300 trees) | Engineered ratio features |
| Meta Learner | MLPClassifier (32→16) | Fuses both probability outputs |

---

## ⚗️ Feature Engineering

| Feature | Formula | Clinical Significance |
|---------|---------|----------------------|
| NLR | Neutrophils / Lymphocytes | Detects infection type |
| PLR | Platelets / WBC | Infection severity marker |
| RBC_HCT | RBC × HCT | Hemoconcentration (plasma leakage) |
| MPV_PLT | MPV × Platelets | Platelet dynamics in dengue |

---

## 📊 Dataset

**Dengue Fever Hematological Dataset**
- Blood count parameters: Platelets, WBC, Neutrophils, Lymphocytes, HCT, RBC, MPV, Haemoglobin, etc.
- Target: `Result` → Dengue Positive / Negative
- Class imbalance handled using **SMOTEENN** (SMOTE + Edited Nearest Neighbours)

---

## 🖥️ App Features

- Medical/clinical UI built with Streamlit
- Patient info input (name, age, gender, fever duration, sample ID)
- Full CBC input with normal range hints
- CBC summary table with Normal / Low ↓ / High ↑ status tags
- Probability donut chart + CBC vs Normal bar chart
- Risk level classification: High / Moderate / Low

---

## 📁 Project Structure

```
dengue-detection/
├── app.py                              # Streamlit web app
├── final (1).py                        # Model training pipeline
├── dengue_simple_model.pkl             # Trained ExtraTrees model
├── Dengue Fever Hematological Dataset.csv  # Dataset
└── README.md
```

---

## ⚕️ Disclaimer

This tool is for **educational and screening purposes only**. It does not replace professional medical diagnosis. All results must be interpreted by a qualified physician. Confirm with NS1 antigen, IgM/IgG serology, and full clinical evaluation.

