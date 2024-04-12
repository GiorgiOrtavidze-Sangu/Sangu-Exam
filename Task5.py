import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load data from spam-data.csv
spam_dataframe = pd.read_csv('spam-data.csv')

# Split data into features (X) and target labels (y)
X_spam = spam_dataframe.drop(columns=['label'])
y_spam = spam_dataframe['label']

# Build and train logistic regression model
spam_model = LogisticRegression()
spam_model.fit(X_spam, y_spam)

# Load email content from emails.txt
with open('emails.txt', 'r') as email_file:
    email_content = email_file.read()

# Extract email features using CountVectorizer
email_vectorizer = CountVectorizer(vocabulary=X_spam.columns)
email_features = email_vectorizer.fit_transform([email_content])

# Classify email as spam or not spam
spam_prediction = spam_model.predict(email_features)
if spam_prediction[0] == 1:
    print("The email is predicted to be spam.")
else:
    print("The email is predicted not to be spam.")

# Analyze feature importance
spam_feature_importance = pd.Series(spam_model.coef_[0], index=X_spam.columns)
print("Feature Importance for Spam Detection:")
print(spam_feature_importance)