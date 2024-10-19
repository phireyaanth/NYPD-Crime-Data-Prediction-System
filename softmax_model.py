# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the cleaned dataset
data = pd.read_csv('cleaned_data_2020_2023.csv')

# Step 1: Remove rows with 'UNKNOWN' in target columns
data = data[
    (data['SUSP_AGE_GROUP'] != 'UNKNOWN') & 
    (data['SUSP_RACE'] != 'UNKNOWN') & 
    (data['SUSP_SEX'] != 'U')  # Ensure 'U' in gender is removed
]

# Prepare the Features and Targets
features = [
    'year', 'month', 'day', 'weekday', 'hour', 'Latitude', 'Longitude',
    'OFNS_DESC', 'ADDR_PCT_CD', 'LAW_CAT_CD', 'BORO_NM', 
    'PREM_TYP_DESC', 'PARKS_NM', 'HADEVELOPT', 'STATION_NAME', 
    'VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX'
]

target_age_group = 'SUSP_AGE_GROUP'
target_race = 'SUSP_RACE'
target_sex = 'SUSP_SEX'

# Step 2: Encoding Categorical Features
categorical_features = [
    'weekday', 'OFNS_DESC', 'LAW_CAT_CD', 'BORO_NM', 
    'PREM_TYP_DESC', 'VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX'
]

# One-Hot Encoding the categorical features
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_cat = encoder.fit_transform(data[categorical_features])

# Concatenate encoded categorical data with the numerical features
numerical_features = [
    'year', 'month', 'day', 'hour', 'Latitude', 'Longitude', 
    'ADDR_PCT_CD', 'PARKS_NM', 'HADEVELOPT', 'STATION_NAME'
]
X = np.hstack([encoded_cat, data[numerical_features]])

# Step 3: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Splitting Data for Each Target Variable
def split_data(X, target):
    return train_test_split(X, data[target], test_size=0.2, random_state=42)

X_train_age, X_test_age, y_train_age, y_test_age = split_data(X_scaled, target_age_group)
X_train_race, X_test_race, y_train_race, y_test_race = split_data(X_scaled, target_race)
X_train_sex, X_test_sex, y_train_sex, y_test_sex = split_data(X_scaled, target_sex)

# Step 5: Building and Training the Logistic Regression Models with Class Weights
def train_and_evaluate_model(X_train, X_test, y_train, y_test, target_name):
    model = LogisticRegression( 
        solver='lbfgs', 
        max_iter=500,
        class_weight='balanced'  # Adjusting class weights here
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\nClassification Report for {target_name}:")
    print(classification_report(y_test, predictions))

# Train and evaluate the model for SUSP_AGE_GROUP
train_and_evaluate_model(X_train_age, X_test_age, y_train_age, y_test_age, target_age_group)

# Train and evaluate the model for SUSP_RACE
train_and_evaluate_model(X_train_race, X_test_race, y_train_race, y_test_race, target_race)

# Train and evaluate the model for SUSP_SEX
train_and_evaluate_model(X_train_sex, X_test_sex, y_train_sex, y_test_sex, target_sex)
