# from datetime import datetime, timedelta
# from geopy.distance import geodesic

# from geopy.geocoders import Nominatim
    
    
# name1=input('Enter start place name:')
# name2=input('Enter end place:')

# mvp=(9.979882, 76.580307)
# kol=(9.9782707, 76.4738971)
# puk=(9.97478605663608,76.4180413861946)
# tri=(9.950012,76.349988)
# vyt=(10.00145,76.2828)
# ekm=(9.981636,76.299884)
# distance_km = geodesic(mvp, ekm).kilometers

# print('Distance:',distance_km)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Sample training dataset
data = pd.DataFrame({
    'ID': range(1, 11),
    'Weather': [25, 28, 22, 20, 30, 26, 27, 23, 21, 29],
    'Traffic': [100, 120, 80, 70, 150, 110, 130, 90, 75, 140],
    'ObservedArrivalTime': [10, 15, 5, 3, 20, 12, 14, 6, 4, 18]
})

# Sample test dataset
new_data = pd.DataFrame({
    'Weather': [27, 24],
    'Traffic': [110, 95],
    'ObservedArrivalTime': [None, None]  # Placeholder for observed arrival time in new data
})

def preprocess_inputs(df):
    df = df.copy()
    # Drop ID column
    df = df.drop('ID', axis=1)
    # Split df into X and y
    y = df['ObservedArrivalTime']
    X = df.drop(['ObservedArrivalTime'], axis=1)
    # Train-test split (for demonstration purposes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    # Model for predicting ETA
    eta_model = RandomForestRegressor(random_state=1)
    eta_model.fit(X_train, y_train)
    return eta_model

def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.2f} minutes")

def predict_eta(model, new_data, scaler):
    # Preprocess the new data
    new_data_input = new_data.drop('ObservedArrivalTime', axis=1)
    new_data_input = pd.DataFrame(scaler.transform(new_data_input), columns=new_data_input.columns)
    
    # Make predictions on the new data
    predicted_eta = model.predict(new_data_input)
    return predicted_eta

# Preprocess the training data
X_train, _, y_train, _, scaler = preprocess_inputs(data)

# Train the model
trained_model = train_model(X_train, y_train)

# Evaluate the model (for demonstration purposes)
evaluate_model(trained_model, X_train, y_train)

# Predict ETA for new test data
predicted_arrival_times = predict_eta(trained_model, new_data, scaler)

# Update the new test data with estimated arrival times
new_data['EstimatedArrivalTime'] = predicted_arrival_times

# Display the new test data with observed and estimated arrival times
print("New Test Data with Observed and Estimated Arrival Times:")
print(new_data)
