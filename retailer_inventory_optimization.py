import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from collections import deque


# Read in Google Sheets file
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("xxxxxx", scope)
client = gspread.authorize(creds)

sheet = client.open("Retail_Data_Updated").sheet1
data = sheet.get_all_values()

df_raw = pd.DataFrame(data[1:], columns=data[0])

# Convert 'Date' column to datetime format
df_raw['Date'] = pd.to_datetime(df_raw['Date'], format='%m/%d/%Y', errors='coerce')

# Function to determine season based on the date
def find_season(date):
    if date.month in [12, 1, 2]:
        return 'Winter'
    elif date.month in [3, 4, 5]:
        return 'Spring'
    elif date.month in [6, 7, 8]:
        return 'Summer'
    elif date.month in [9, 10, 11]:
        return 'Fall'

# Create 'Season' column based on 'Date'
df_raw['Season'] = df_raw['Date'].apply(find_season)

df = df_raw.copy()

# Perform one-hot encoding of categorical variables
df = pd.get_dummies(df, columns=['Product_Description', 'Product_Category', 'Product_Line', 'Raw_Material', 'Region'], drop_first=True)

# Prepare data for the classifier and split into train and test sets
x_class = df.drop(columns=['Date', 'Customer_ID', 'Product_ID', 'Latitude','Longitude', 'Season'])
y_class = df['Season']

# Convert season labels to numeric values
season_mapping = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}
y_class = y_class.map(season_mapping)

x_train_class, x_test_class, y_train_class, y_test_class = train_test_split(x_class, y_class, test_size=0.3, random_state=42)

# Train the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train_class, y_train_class)

# Predict on the test set
y_pred_class = clf.predict(x_test_class)
print(f'Classifier Accuracy: {accuracy_score(y_test_class, y_pred_class)}')

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(x_train_class, y_train_class)
print(f'Best parameters: {grid_search.best_params_}')

best_rf_model = RandomForestClassifier(bootstrap=True, max_depth=20, min_samples_leaf=2,
                                      min_samples_split=10, n_estimators=200, random_state=42)
best_rf_model.fit(x_train_class, y_train_class)
y_pred = best_rf_model.predict(x_test_class)
accuracy = accuracy_score(y_test_class, y_pred)
print(f"Optimized RandomForest Accuracy: {accuracy:.2f}")
