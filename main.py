import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load Dataset
fake = pd.read_csv('data/Fake.csv')
true = pd.read_csv('data/True.csv')

fake['label'] = 0  # Fake
true['label'] = 1  # Real

data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

X = data['text']
y = data['label']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_vec, y_train)

# Accuracy
score = accuracy_score(y_test, model.predict(X_test_vec))
print(f"Accuracy: {score * 100:.2f}%")

# Save Model & Vectorizer
joblib.dump(model, 'model/fake_news_model.pkl')
joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')
