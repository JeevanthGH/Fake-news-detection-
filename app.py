from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load('model/fake_news_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    news_vec = vectorizer.transform([news])
    prediction = model.predict(news_vec)

    output = "Real News" if prediction[0] == 1 else "Fake News"
    return render_template('index.html', prediction_text=output)

if __name__ == '__main__':
    app.run(debug=True)
