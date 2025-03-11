import streamlit as st

def main():
    st.title("Model Development Guidelines")
    st.write("""
    For our machine learning models, we employ Support Vector Machine (SVM) and K-Nearest Neighbors (KNN) to predict disease outcomes based on medical data
             """)
    st.subheader("Implementation of SVM")
    st.write("""
    The Support Vector Machine (SVM) model is implemented using the scikit-learn library. The following steps are taken :
             """) 
    st.write("1. Instantiate the SVM model :")
    st.code("""
        from sklearn.svm import SVC
        svm_model = SVC(kernel = 'linear', C = 1.0, gamma = 'scale')
        """, language='python')
    st.write("2. Train the model on the training set :")
    st.code("""
        svm_model.fit(X_train, y_train)
        """, language='python')
    st.write("3. Make predictions on the test set :")
    st.code("""
        y_pred = svm_model.predict(X_test)
        """, language='python')
    st.write("4. Evaluate the model performance :")
    st.code("""
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(y_test, y_pred)
        print("SVM Accuracy:", accuracy)
        print(classification_report(y_test, y_pred))
        """, language='python')
    

    st.subheader("Implementation of KNN")
    st.write("""
    The K-Nearest Neighbors (KNN) model is implemented as follows :
             """) 
    st.write("1. Instantiate the KNN classifier with k = 5 :")
    st.code("""
        from sklearn.neighbors import KNeighborsClassifier
        knn_model = KNeighborsClassifier(n_neighbors = 5)
        """, language='python')
    st.write("2. Train the model :")
    st.code("""
        knn_model.fit(X_train, y_train)
        """, language='python')
    st.write("3. Make predictions :")
    st.code("""
        y_pred = knn_model.predict(X_test)
        """, language='python')
    st.write("4. Evaluate the model :")
    st.code("""
        accuracy = accuracy_score(y_test, y_pred)
        print("KNN Accuracy:", accuracy)
        print(classification_report(y_test, y_pred))
        """, language='python')
    
    st.header("Neural Network Model Development")
    st.write("For our deep learning approach, we use a feedforward neural network built with TensorFlow and Keras")
    st.subheader("Neural Network Architecture")
    st.write("""
    The neural network consists of :
    - Input Layer : Takes preprocessed features as input.
    - Hidden Layers : Uses ReLU activation for non-linearity.
    - Output Layer : Uses Sigmoid activation for binary classification.
             """)
    st.subheader("Model Implementation")
    st.write("""
    The K-Nearest Neighbors (KNN) model is implemented as follows :
             """) 
    st.write("1. Define the model architecture :")
    st.code("""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense

        nn_model = Sequential([
            Dense(16, activation = 'relu', input_shape=(X_train.shape[1],)),
            Dense(8, activation = 'relu'),
            Dense(1, activation = 'sigmoid')
        ])
        """, language='python')
    st.write("2. Compile the model :")
    st.code("""
        nn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        """, language='python')
    st.write("3. Train the model :")
    st.code("""
        nn_model.fit(X_train, y_train, epochs = 50, batch_size = 16, validation_data = (X_test, y_test))
        """, language='python')
    st.write("4. Evaluate the model :")
    st.code("""
        loss, accuracy = nn_model.evaluate(X_test, y_test)
        print("Neural Network Accuracy:", accuracy)
        """, language='python')
    
    st.subheader("Summary of Development Steps")
    st.write("""
    1. Theoretical Background
    - Explanation of SVM and KNN concepts.
    2. Data Preparation
    - Handling missing values
    - Encoding categorical data
    - Feature scaling
    - Splitting dataset into training/testing sets
    3. Machine Learning Model (SVM & KNN)
    - Implemented SVM with RBF kernel
    - Implemented KNN with k=5 neighbors
    - Evaluated models using accuracy, precision, and recall
    4. Neural Network Model
    - Designed a feedforward neural network with ReLU activation
    - Trained using Adam optimizer with binary cross-entropy loss
    - x Evaluated model accuracy and loss performance
             """)
    
main()