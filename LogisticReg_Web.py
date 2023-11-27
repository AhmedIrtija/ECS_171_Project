import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_array

# Load the dataset
data = pd.read_csv('Allergen_Status_of_Food_Products.csv')

# Check for and handle NaN values in the 'Prediction' column
data_cleaned = data.dropna(subset=['Prediction'])

# Dropping unnecessary columns
data_cleaned = data_cleaned.drop(['Price ($)', 'Customer rating (Out of 5)', 'Allergens'], axis=1)

# Encoding categorical variables
# Modify encoder initialization to handle unknown categories
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(data_cleaned[['Food Product', 'Main Ingredient', 'Sweetener', 'Fat/Oil', 'Seasoning']])
feature_names = encoder.get_feature_names_out(['Food Product', 'Main Ingredient', 'Sweetener', 'Fat/Oil', 'Seasoning'])

# Split the data
X = X_encoded
y = data_cleaned['Prediction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Create and train the logistic regression model
logreg = LogisticRegression(max_iter=10000, penalty='l2')
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

# Calculate precision and recall for 'Contains'
precision_contains = precision_score(y_test, y_pred, pos_label='Contains')
precision_does_not_contain = precision_score(y_test, y_pred, pos_label='Does not contain')
recall_contains = recall_score(y_test, y_pred, pos_label='Contains')
recall_does_not_contain = recall_score(y_test, y_pred, pos_label='Does not contain')

# Print the precision and recall
print("Precision of 'Contains':", precision_contains)
print("Recall of 'Contains':", recall_contains)
print("Precision of 'Does not contain':", precision_does_not_contain)
print("Recall of 'Does not contain':", recall_does_not_contain)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print("Accuracy", accuracy)
print("Classification Report")
print(classification_rep)

with open("Allergen.pkl", "wb") as file:
    pickle.dump(logreg, file)
