import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from collections import deque


# Read in Google Sheets file
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("retailer-optimization-45f5b4d693ba.json", scope)
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
x_class = df.drop(columns=['Date', 'Customer_ID', 'Product_ID', 'Latitude','Longitude', 'Season', 'Date_Unformatted'])
y_class = df['Season']

# Convert season labels to numeric values
season_mapping = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}
y_class = y_class.map(season_mapping)

x_train_class, x_test_class, y_train_class, y_test_class = train_test_split(x_class, y_class, test_size=0.3, random_state=42)

model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
model3 = SVC(probability=True, random_state=42)

meta_model = LogisticRegression()

stacked_model = VotingClassifier(
    estimators=[
        ('rf', model1),
        ('gb', model2),
        ('svc', model3)
    ],
    voting='soft'
)

stacked_model.fit(x_train_class, y_train_class)

y_pred = stacked_model.predict(x_test_class)
accuracy = accuracy_score(y_test_class, y_pred)
print(f"Stacked Model Accuracy: {accuracy:.2f}")



from sklearn.model_selection import GridSearchCV

# Define parameter grids for each model
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [3, 5, 7]
}

param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Initialize models
model1 = RandomForestClassifier(random_state=42)
model2 = GradientBoostingClassifier(random_state=42)
model3 = SVC(probability=True, random_state=42)

# Initialize GridSearchCV for each model
grid_search_rf = GridSearchCV(estimator=model1, param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2)
grid_search_gb = GridSearchCV(estimator=model2, param_grid=param_grid_gb, cv=3, n_jobs=-1, verbose=2)
grid_search_svc = GridSearchCV(estimator=model3, param_grid=param_grid_svc, cv=3, n_jobs=-1, verbose=2)

grid_search_rf.fit(x_train_class, y_train_class)
grid_search_gb.fit(x_train_class, y_train_class)
grid_search_svc.fit(x_train_class, y_train_class)

print(f'Best parameters for RF: {grid_search_rf.best_params_}')
print(f'Best parameters for GB: {grid_search_gb.best_params_}')
print(f'Best parameters for SVC: {grid_search_svc.best_params_}')

best_rf = grid_search_rf.best_estimator_
best_gb = grid_search_gb.best_estimator_
best_svc = grid_search_svc.best_estimator_

stacked_model = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('gb', best_gb),
        ('svc', best_svc)
    ],
    voting='soft'
)

stacked_model.fit(x_train_class, y_train_class)

y_pred = stacked_model.predict(x_test_class)
accuracy = accuracy_score(y_test_class, y_pred)
print(f"Stacked Model Accuracy with Best Parameters: {accuracy:.2f}")
