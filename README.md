# Fake News Detection

## Overview
This project implements a Fake News Detection system using a deep learning model. It classifies news articles as either "Real" or "Fake" based on text content. The model is built using LSTM (Long Short-Term Memory) networks and is trained on a dataset of true and fake news articles.

## Features
- Cleans and preprocesses news articles using NLP techniques
- Trains an LSTM-based deep learning model for classification
- Loads an existing model if available, or trains a new one
- Supports input from text, file, or URL
- Fetches and processes news content from a given URL
- Generates a confusion matrix for model evaluation

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install pandas numpy tensorflow matplotlib seaborn scikit-learn nltk beautifulsoup4 requests
```

## Dataset
The model is trained on the Fake News Dataset, which includes:
- `Fake.csv` - containing fake news articles
- `True.csv` - containing real news articles

Both datasets must be placed inside the `Fake_news_Detection/` directory.

## How It Works
1. **Data Preprocessing:**
   - Converts text to lowercase
   - Removes numbers and special characters
   - Applies lemmatization and stopword removal
   - Tokenizes and pads sequences for deep learning input

2. **Model Training:**
   - Uses an LSTM model with embedding layers, dropout, and dense layers
   - Splits the dataset into training and validation sets
   - Trains for 5 epochs with a batch size of 32
   - Saves the trained model as `fake_news_model.h5`

3. **Prediction:**
   - Loads the pre-trained model (if available)
   - Accepts input as raw text, a file, or a URL
   - Extracts text and classifies it as Real or Fake

## Usage
### Running the Program
Run the script to start the Fake News Detection system:

```bash
python fake_news_detection.py
```

### Input Methods
- Enter text directly
- Provide a file path containing news content
- Provide a URL to fetch news content

### Example Output
```
Enter a news article (text, file path, or URL) or type 'exit' to stop:
https://example.com/fake-news-article

Checking the URL...
Prediction probability: 0.12
This news is FAKE
```

## Model Evaluation
The script also generates a confusion matrix for performance analysis:

```python
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

## Notes
- If `fake_news_model.h5` exists, the model loads automatically.
- If missing, the model will be trained from scratch.
- The dataset should be placed in `Fake_news_Detection/`.

## License
This project is open-source and available for modification and enhancement.
