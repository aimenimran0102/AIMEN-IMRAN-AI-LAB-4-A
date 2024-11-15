# AIMEN-IMRAN-AI-LAB-4-A
HOME TASK CODE:
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Sample data (Manually converted from the image)
data = {
    'Product ID': [301, 302, 303, 304, 305, 306],
    'Product Category': ['Electronics', 'Clothing', 'Food', 'Furniture', 'Accessories', 'Electronics'],
    'Units Sold': [500, 150, 50, 400, 100, 300],
    'Price': [100, 30, 10, 200, 20, 120],
    'Seasonal Demand': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes'],
    'Supplier Reliability': [5, 3, 4, 4, 2, 5],
    'Return Rate (%)': [2, 5, 3, 1, 8, 3],
    'Stock Availability': ['High', 'Medium', 'Low', 'High', 'Medium', 'High'],
    'Demand Category': ['High Demand', 'Moderate Demand', 'Low Demand', 'High Demand', 'Moderate Demand', 'High Demand']
}

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Encoding categorical features
le = preprocessing.LabelEncoder()
df['Product Category'] = le.fit_transform(df['Product Category'])
df['Seasonal Demand'] = le.fit_transform(df['Seasonal Demand'])
df['Stock Availability'] = le.fit_transform(df['Stock Availability'])
df['Demand Category'] = le.fit_transform(df['Demand Category'])

# Selecting features and label
features = df[['Product Category', 'Units Sold', 'Price', 'Seasonal Demand', 
               'Supplier Reliability', 'Return Rate (%)', 'Stock Availability']]
label = df['Demand Category']

# Splitting the data
features_train, features_test, label_train, label_test = train_test_split(features, label, test_size=0.2, random_state=42)

# Initialize and train the model
model = GaussianNB()
model.fit(features_train, label_train)

# Make predictions
predicted = model.predict(features_test)
print("Predicted Result:", predicted)

# Calculate confusion matrix and accuracy
conf_mat = confusion_matrix(label_test, predicted)
print("Confusion Matrix:\n", conf_mat)

accuracy = accuracy_score(label_test, predicted)
print("Accuracy:", accuracy)

