import streamlit as st

def main():
    st.title("Data Preperation & Algorithms")
    
    st.header("1. Data Preparation")
    st.write("""
    Before training our models, we need to preprocess the dataset to ensure that the data is clean, structured, and suitable for machine learning algorithms. The key steps involved in data preparation are as follows:
            """)
    
    st.subheader("1.0 Sorce of Data")
    st.write("- Heart Disease Dataset : Kaggle")
    st.write("- MNIST : Tensorflow Dataset")

    st.subheader("1.1 Handling Missing Values")
    st.write("""
    Real-world datasets often contain missing values, which can affect model performance. To handle missing data, we apply the following techniques:
    - Mean/Median Imputation : Replace missing numerical values with the mean or median of the column.
    - Mode Imputation : Replace missing categorical values with the most frequent category.
    - Removing Rows : If missing values are significant, we may choose to remove those rows entirely.
             """)
    
    st.subheader("1.2 Encoding Categorical Data")
    st.write("""
    Machine learning models work with numerical data, so categorical features must be converted into numerical values using:
    - One-Hot Encoding : Convert categorical variables into binary vectors.
    - Label Encoding : Assign a unique integer to each category.
             """)
    
    st.subheader("1.3 Feature Scaling")
    st.write("""
    Since some machine learning models are sensitive to feature magnitudes, we apply scaling techniques such as:
    - Standardization : Transform features to have zero mean and unit variance.
    - Normalization : Scale values between 0 and 1 to ensure uniformity across features.
             """)
    
    st.subheader("1.4 Splitting the Dataset")
    st.write("""
    The dataset is divided into training and testing sets to evaluate model performance properly.
             """)
    
    st.header("Theory of Algorithms")
    st.subheader("1.1 Support Vector Machine (SVM)")
    st.write("""
    Support Vector Machine (SVM) is a supervised learning algorithm used for classification and regression. The key idea behind SVM is to find an optimal hyperplane that separates data points of different classes with the maximum margin. The main concepts include:
    - Hyperplane : A decision boundary that best separates the classes.
    - Support Vectors : Data points that lie closest to the hyperplane and influence its position.
    - Margin : The distance between the hyperplane and the closest support vectors. A larger margin improves generalization.
    - Kernel Trick : A technique that transforms non-linearly separable data into a higher-dimensional space where it becomes linearly separable. Common kernels include linear, polynomial, and radial basis function (RBF).
             """)
    
    st.subheader("1.2 K-Nearest Neighbors (KNN)")
    st.write("""
    K-Nearest Neighbors (KNN) is a simple and intuitive non-parametric algorithm used for classification. It works by comparing new data points to the existing labeled data based on distance metrics such as:
    - Euclidean Distance : Measures straight-line distance between points.
    - Manhattan Distance : Measures distance along the grid lines.
    - Minkowski Distance : Generalized form of Euclidean and Manhattan distances.
             """)
    
    st.subheader("1.3 Neural Networks")
    st.write("""
        Neural Networks are a subset of machine learning inspired by the structure and function of the human brain. They consist of layers of interconnected nodes (neurons) that process and learn patterns from data. The fundamental components of neural networks include:
- Neurons (Nodes) : The basic computational unit that takes an input, applies a weight, and passes it through an activation function.
- Layers : Neural networks are organized into different layers:\n
        Input Layer : Receives input features.
        Hidden Layers : Perform computations and extract features.
        Output Layer : Produces the final prediction.
- Weights and Biases : Parameters that determine how much influence a neuron has on the next layer.
- Activation Functions : Non-linear functions applied to each neuron to introduce non-linearity, allowing the network to learn complex patterns. Common activation functions include :\n
        ReLU (Rectified Linear Unit): f(x) = max(0, x) (widely used due to simplicity and efficiency).
        Sigmoid: f(x) = 1 / (1 + e^(-x)) (commonly used in binary classification).
        Softmax: Used for multi-class classification.
- Loss Function: Measures how well the neural network's predictions match the actual values. For classification tasks, common loss functions include:\n
        Binary Cross-Entropy : Used for binary classification.
        Categorical Cross-Entropy : Used for multi-class classification.
- Backpropagation and Gradient Descent :\n
        Backpropagation : An algorithm that calculates the gradient of the loss function with respect to each weight in the network.
        Gradient Descent : Optimizes weights by adjusting them in the direction that minimizes the loss function.
Neural networks are particularly effective for tasks like image recognition, natural language processing, and complex pattern recognition.   
             """)
    
main()
