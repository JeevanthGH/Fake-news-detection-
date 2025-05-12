# Fake News Detection using NLP and Machine Learning with Streamlit UI

import os
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

# Load dataset or create a sample one if not found
DATA_PATH = "fake_news_dataset.csv"
if not os.path.exists(DATA_PATH):
    print(f"Warning: {DATA_PATH} not found. Creating a sample dataset for demonstration.")
    data = pd.DataFrame({
        'text': [
            'The moon landing was faked by NASA',
            'Water boils at 100 degrees Celsius',
            'COVID-19 was created in a lab',
            'The Earth orbits the Sun',
            'Vaccines cause autism'
        ],
        'label': [1, 0, 1, 0, 1]  # 1 = Fake, 0 = Real
    })
else:
    data = pd.read_csv(DATA_PATH)

# Preprocessing function
custom_stopwords = ENGLISH_STOP_WORDS

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in custom_stopwords]
    return ' '.join(words)

# Apply preprocessing
data['text'] = data['text'].apply(preprocess_text)

# Show class distribution
print("Label distribution:", Counter(data['label']))

# Feature extraction with n-gram support
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(data['text']).toarray()
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Streamlit App
st.title("Fake News Detection App")
st.write("Enter a news statement to classify it as real or fake.")

user_input = st.text_area("News Text", "The government cloned dinosaurs for war purposes.")

if st.button("Classify"):
    cleaned_input = preprocess_text(user_input)
    vector_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(vector_input)[0]
    st.write(f"### Prediction: {'Fake' if prediction == 1 else 'Real'}")
