import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocess import combine_and_clean

print("Step 1: Loading data...")
df = pd.read_csv('data/fake_job_postings.csv')
print(f"Loaded {len(df)} job postings")

print("\nStep 2: Cleaning text...")
df = combine_and_clean(df)
print("Text cleaning done!")

# Separate features (X) and labels (y)
# Why: X is what the model reads, y is what it needs to predict
X = df['cleaned_text']
y = df['fraudulent']

print("\nStep 3: Splitting data into train and test sets...")
# Why: we train on 80% and test on the remaining 20% the model has never seen
# stratify=y means fake/real ratio stays the same in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print("\nStep 4: Converting text to numbers using TF-IDF...")
# Why: ML models only understand numbers, not words
# max_features=10000 means we keep only 10,000 most important words
# ngram_range=(1,2) means we also look at 2-word phrases like "no experience"
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

# fit_transform on TRAIN only — model learns vocabulary from training data
X_train_tfidf = tfidf.fit_transform(X_train)
# transform only on TEST — we don't let model peek at test data vocabulary
X_test_tfidf = tfidf.transform(X_test)
print("TF-IDF done! Each job is now a row of 10,000 numbers")

print("\nStep 5: Training Logistic Regression model...")
# Why class_weight='balanced': only 5% jobs are fake
# without this, model ignores fake jobs and still gets 95% accuracy (useless)
# balanced forces model to pay equal attention to both classes
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_tfidf, y_train)
print("Model trained!")

print("\nStep 6: Evaluating model...")
y_pred = model.predict(X_test_tfidf)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

print("\nStep 7: Saving model and vectorizer...")
# Why: we save the trained model so our Streamlit app can load it
# without saving, we'd have to retrain every time which takes minutes
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl')
joblib.dump(tfidf, 'model/tfidf.pkl')
print("Model saved to model/model.pkl")
print("TF-IDF saved to model/tfidf.pkl")

print("\nStep 8: Saving confusion matrix image...")
# Why: this image shows exactly where model is right and wrong
# great to include in your GitHub README for interviews
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])
plt.title('HireGuard AI - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('model/confusion_matrix.png')
print("Confusion matrix saved to model/confusion_matrix.png")
print("\nDay 2 Complete! Model is trained and saved.")