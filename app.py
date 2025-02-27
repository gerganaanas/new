
import streamlit as st
import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

path = kagglehub.dataset_download("uciml/default-of-credit-card-clients-dataset")

print("Path to dataset files:", path)
df = pd.read_csv("UCI_Credit_Card.csv")

# Drop 'ID' as it is not useful for prediction
df = df.drop(columns=['ID'])


# Feature Selection: Keep only relevant features
selected_features = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'BILL_AMT1', 'BILL_AMT2', 'PAY_AMT1', 'PAY_AMT2']
X = df[selected_features]
y = df['default.payment.next.month']

# Split Data into Train and Test Sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize Data for kNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train kNN Classifier
def train_knn(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print("kNN Classifier Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_knn))
    print(classification_report(y_test, y_pred_knn))
    return knn

knn_model = train_knn(X_train_scaled, X_test_scaled, y_train, y_test)


# Function for User Input Prediction
def predict_default():
    print("Enter the following details to predict default risk:")
    input_data = []
    input_data.append(float(input("LIMIT_BAL: ")))
    input_data.append(int(input("AGE: ")))
    input_data.append(int(input("PAY_0 (Repayment status for Sep): ")))
    input_data.append(int(input("PAY_2 (Repayment status for Aug): ")))
    input_data.append(int(input("PAY_3 (Repayment status for Jul): ")))
    input_data.append(float(input("BILL_AMT1 (Recent bill statement): ")))
    input_data.append(float(input("BILL_AMT2 (Previous bill statement): ")))
    input_data.append(float(input("PAY_AMT1 (Recent payment amount): ")))
    input_data.append(float(input("PAY_AMT2 (Previous payment amount): ")))

    # Print out user input values
    print("\nUser Input Values:")
    feature_names = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'BILL_AMT1', 'BILL_AMT2', 'PAY_AMT1', 'PAY_AMT2']
    for name, value in zip(feature_names, input_data):
        print(f"{name}: {value}")
    
    # Convert to NumPy array and reshape
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)  # Scale input for kNN
    
    # Make Predictions
    knn_pred = knn_model.predict(input_scaled)
    
    print("\nPrediction Results:")
    print(f"kNN Model Prediction: {'Default' if knn_pred[0] == 1 else 'No Default'}")


# Call the function to test with user input
predict_default()  # Runs automatically