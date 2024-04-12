import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load data from the text file
spam_data = np.loadtxt('spam-data.csv')

# Separate features (X) and target labels (y)
X_spam = spam_data[:, :-1]
y_spam = spam_data[:, -1]

# Split the data into training and testing sets
X_train_spam, X_test_spam, y_train_spam, y_test_spam = train_test_split(X_spam, y_spam, test_size=0.2, random_state=42)

# Build and train the logistic regression model
spam_model = LogisticRegression()
spam_model.fit(X_train_spam, y_train_spam)

# Evaluate the model on the testing data
y_pred_spam = spam_model.predict(X_test_spam)

# Print the confusion matrix
conf_matrix_spam = confusion_matrix(y_test_spam, y_pred_spam)
print("Confusion Matrix for Spam Detection:")
print(conf_matrix_spam)