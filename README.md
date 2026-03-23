# HireGuard AI — Fake Job Detection System 🤖

## 📌 Overview

HireGuard AI is a machine learning-based system that detects fraudulent job postings using Natural Language Processing (NLP). It helps users identify scams in online job listings.

---

## 🎯 Problem Statement

Fake job postings are increasing on platforms like LinkedIn and Indeed, leading to fraud and data theft.

---

## 💡 Solution

This system classifies job descriptions as:

* ✅ Real
* ❌ Fake

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* TF-IDF Vectorizer
* Streamlit (UI)

---

## 🧠 Model Architecture

Raw Text
↓
Text Preprocessing
↓
Feature Extraction (TF-IDF)
↓
ML Model (Logistic Regression / Naive Bayes)
↓
Prediction (Real / Fake)

---

## 📂 Folder Structure

```
hireguard-ai/
│── data/
│── notebooks/
│── model/
│── app/
│── utils/
│── main.py
```

---

## ⚙️ Installation

1. Clone:

```
git clone https://github.com/your-username/hireguard-ai.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run app:

```
streamlit run app.py
```

---

## 📊 Dataset

* Fake Job Posting Dataset (Kaggle)

---

## 📈 Features

* Text classification
* Confidence score
* Highlight suspicious keywords

---

## 🧠 Key Learnings

* NLP fundamentals
* Feature engineering (TF-IDF)
* Model evaluation (Accuracy, Precision, Recall)
* Deployment using Streamlit

---

## 🚀 Future Scope

* Use BERT for better accuracy
* Browser extension for real-time detection
* Integration with job platforms

---

## 📸 Demo

(Add GIF/video later)

---

## 🤝 Contribution

Suggestions and improvements are welcome!
