import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

st.title("SVM & KNN Model")

# Load Dataset
data = pd.read_csv("dataset/heart.csv")

# Auto EDA using ydata-profiling
st.subheader("📊 Automated EDA Report ( Heart Disease Dataset )")
profile = ProfileReport(data, explorative=True)
st_profile_report(profile)

# One-Hot Encoding
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Select Features
X = data[['age', 'thalach']].values
y = data['target'].values

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train Models
svm_model = SVC(kernel='linear', C=1.0)
knn_model = KNeighborsClassifier(n_neighbors=5)

svm_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

# Test Models
y_pred_svm = svm_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# Plot Decision Boundary
def plot_decision_boundary(model, X, y, title, input_point=None):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm')
    
    if input_point is not None:
        plt.scatter(input_point[0], input_point[1], color='yellow', edgecolors='black', s=200, label="New Data")
        plt.legend()

    plt.xlabel('Age (scaled)')
    plt.ylabel('Thalach (scaled)')
    plt.title(title)

    return plt

# Streamlit UI
st.write(f'### 🎯 SVM Accuracy: {accuracy_svm:.2f}')
st.text(classification_report(y_test, y_pred_svm))

st.write(f'### 🎯 KNN Accuracy: {accuracy_knn:.2f}')
st.text(classification_report(y_test, y_pred_knn))

# Demo predictions form
st.subheader("🩺 Predict Heart Disease")
age = st.number_input("Enter Age", min_value=20, max_value=100, step=1)
thalach = st.number_input("Enter Max Heart Rate (Thalach)", min_value=60, max_value=220, step=1)
model_choice = st.radio("Choose Model", ("SVM", "KNN"))

if st.button("🔮 Predict"):
    input_data = np.array([[age, thalach]])
    input_scaled = scaler.transform(input_data)

    if model_choice == "SVM":
        prediction = svm_model.predict(input_scaled)
        model_used = "Support Vector Machine (SVM)"
        model_accuracy = accuracy_svm
        fig = plot_decision_boundary(svm_model, X_scaled, y, "SVM Decision Boundary", input_point=input_scaled[0])
    else:
        prediction = knn_model.predict(input_scaled)
        model_used = "K-Nearest Neighbors (KNN)"
        model_accuracy = accuracy_knn
        fig = plot_decision_boundary(knn_model, X_scaled, y, "KNN Decision Boundary", input_point=input_scaled[0])

    result_text = "🔴 High Risk (Heart Disease)" if prediction[0] == 1 else "🟢 Low Risk (No Heart Disease)"
    
    st.success(f"**Prediction using {model_used}:** {result_text}")
    st.write(f"📊 **Model Accuracy:** {model_accuracy:.2f}")

    # Show Graph with New Data Point
    st.pyplot(fig)
