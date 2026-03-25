# 🛡️ HireGuard AI — Fake Job Detection System

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)

## 🔴 Live Demo
👉 **[Try HireGuard AI Live](https://hireguard-ai-rycyp56qemupft4gh4jekf.streamlit.app)**

## 📌 Overview
HireGuard AI is a machine learning system that detects fraudulent 
job postings using Natural Language Processing (NLP). 
It helps job seekers identify scams before they become victims.

## 🎯 Problem
Fake job postings are increasing on platforms like LinkedIn and 
Indeed, leading to financial fraud and personal data theft.

## 💡 Solution
Paste any job posting and get instant AI analysis:
- ✅ Real or Fake prediction
- 📊 Confidence score
- 🔎 Suspicious keyword detection
- 💡 Safety tips

## 🛠️ Tech Stack
- **Python** — core language
- **Pandas & NumPy** — data processing
- **Scikit-learn** — ML model (Logistic Regression)
- **TF-IDF Vectorizer** — text to numbers
- **NLTK** — text preprocessing
- **Streamlit** — web app deployment

## 🧠 How It Works
```
Raw Job Text
     ↓
Text Cleaning (lowercase, remove URLs, stopwords)
     ↓
TF-IDF Vectorization (text → 10,000 numbers)
     ↓
Logistic Regression Model
     ↓
Prediction → Real / Fake + Confidence Score
```

## 📊 Model Performance
| Metric | Real | Fake |
|--------|------|------|
| Precision | 0.99 | 0.79 |
| Recall | 0.97 | 0.92 |
| F1-Score | 0.98 | 0.85 |

> Fake Recall of 92% means the model catches 92 out of 
> every 100 fake job postings.

## 📂 Project Structure
```
hireguard-ai/
├── data/                    # dataset
├── model/                   # saved model files
├── notebooks/               # exploration notebook
├── utils/
│   └── preprocess.py        # text cleaning functions
├── train.py                 # model training script
├── app.py                   # Streamlit web app
└── requirements.txt         # dependencies
```

## ⚙️ Run Locally
```bash
git clone https://github.com/theshubhamtripathi/hireguard-ai.git
cd hireguard-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## 🚀 Future Improvements
- BERT model for better semantic understanding
- Browser extension for real-time detection
- Integration with LinkedIn and Indeed APIs

---
Made with ❤️ by Shubham Tripathi