# SPAM-SMS-DETECTION

Spam SMS Detection - README

Task Objective

The goal of this project is to develop a robust machine learning model that can accurately classify SMS messages as either Spam or Not Spam (Ham). The dataset consists of labeled messages, and the model uses Natural Language Processing (NLP) techniques for preprocessing and machine learning algorithms for classification.

Features

Text Preprocessing: Cleaning, tokenization, and removal of stopwords.

Feature Extraction: TF-IDF Vectorization to convert text into numerical format.

Model Training: Naïve Bayes classifier for spam detection.

Evaluation: Accuracy, Precision, Recall, and F1-score.

Real-time SMS Prediction: Users can input new SMS messages and check if they are spam or not.

Steps to Run the Project

1. Setup the Environment

Open Google Colab (or run locally in Jupyter Notebook).

Upload the dataset (spam.csv).

Install necessary Python libraries:

!pip install numpy pandas scikit-learn nltk joblib

2. Execute the Training Script

Run spam_sms_detection.py to:

Load and preprocess the dataset.

Train the Naïve Bayes model.

Evaluate performance metrics.

Save the trained model (spam_classifier.pkl) and TF-IDF vectorizer (tfidf_vectorizer.pkl).

3. Classify a New SMS Message

Run predict_sms.py (or use the script in Google Colab) to:

Input a new SMS message.

Preprocess the text.

Use the trained model to predict whether it's Spam or Not Spam.

Enter the SMS text: "Congratulations! You won a free iPhone! Claim now."
Prediction: Spam

4. (Optional) Deploy the Model

Convert the script into a Flask API for real-time SMS classification.

Deploy using Google Cloud, AWS, or Hugging Face Spaces.

