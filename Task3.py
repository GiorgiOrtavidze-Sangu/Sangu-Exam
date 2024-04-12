import numpy as np
import pandas as pd

# Load data from the text file
# Assuming 'spam-data.csv' contains the dataset
spam_data = np.loadtxt('spam-data.csv')

# Display the shape and first few rows of the loaded data
print("Shape of the data:", spam_data.shape)
print("First few rows of the data:\n", spam_data[:5])

# Load data from the text file into a DataFrame
# Assuming 'spam-data.txt' contains the dataset and it's tab-separated
spam_dataframe = pd.read_csv('spam-data.txt', delimiter='\t')

# Display the shape and first few rows of the loaded data
print("Shape of the data:", spam_dataframe.shape)
print("First few rows of the data:\n", spam_dataframe.head())

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# Load the trained logistic regression model
# Assuming the model is already trained and available
spam_model = LogisticRegression()
spam_model.coef_ = np.array([[-0.5]])  # Example coefficients, replace with actual values
spam_model.intercept_ = np.array([-0.1])  # Example intercept, replace with actual values

# Load the email data from the "emails.txt" file
# Assuming 'emails.txt' contains email content
with open('emails.txt', 'r') as email_file:
    email_content = email_file.read()

# Preprocess the email data (e.g., extract features)
# Assuming we use simple Bag-of-Words representation
email_vectorizer = CountVectorizer()
email_features = email_vectorizer.fit_transform([email_content])

# Predict whether the email is spam or not
spam_prediction = spam_model.predict(email_features)

# Print the prediction result on the console
if spam_prediction[0] == 1:
    print("The first email is predicted to be spam.")
else:
    print("The first email is predicted not to be spam.")