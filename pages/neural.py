import streamlit as st
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import gdown
import matplotlib.pyplot as plt
from PIL import Image

st.title("Neural Network Model")

# Google Drive File ID from the model from Colab
drive_link = "https://drive.google.com/uc?id=1DdgsCPeSFxPfaAW7c5gCurXzrxPrZiHZ"
model_path = "mnist_cnn.h5"

gdown.download(drive_link, model_path, quiet=False)

# Load the model
model = tf.keras.models.load_model(model_path)

# Load MNIST Dataset
(ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)

# Function to normalize data
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

ds_test = ds_test.map(normalize_img)

# Display examples of data
st.subheader("📊 Sample Images from MNIST")

fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for ax, (image, label) in zip(axes, ds_train.take(5)):
    ax.imshow(image.numpy().squeeze(), cmap='gray')
    ax.set_title(f"Label: {label.numpy()}")
    ax.axis('off')

st.pyplot(fig)

st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #626567;
        color: white;
        font-size: 18px;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #17202a;
    }
    </style>
""", unsafe_allow_html=True)

# Evaluate the model
if st.button("📈 Evaluate Model"):
    ds_test_batch = ds_test.batch(32)
    test_loss, test_acc = model.evaluate(ds_test_batch)
    st.write(f"### 🎯 Test Accuracy : {test_acc:.4f}")

# Test the model with random data
st.subheader("🖼️ Test Model with Random Image")

def predict_random_image():
    for image, label in ds_test.shuffle(1000).take(1):  
        image_np = image.numpy()
        label_true = label.numpy()

        if image_np.ndim == 2: 
            image_np = np.expand_dims(image_np, axis=-1)
        
        image_reshaped = np.expand_dims(image_np, axis=0)

        # Predict
        pred = model.predict(image_reshaped)
        pred_label = np.argmax(pred)

        # Display
        fig, ax = plt.subplots()
        ax.imshow(image_np.squeeze(), cmap='gray')
        ax.set_title(f"True: {label_true}, Predicted: {pred_label}")
        ax.axis('off')

        st.pyplot(fig)

if st.button("🔮 Predict Random Image"):
    predict_random_image()

# Upload file to predict
st.subheader("📤 Upload Image for Prediction")

uploaded_file = st.file_uploader("Choose an image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))

    st.image(image, caption="Uploaded Image (28x28)", use_column_width=False)

    image_np = np.array(image) / 255.0
    image_np = np.expand_dims(image_np, axis=-1)
    image_np = np.expand_dims(image_np, axis=0)

    pred = model.predict(image_np)
    pred_label = np.argmax(pred)

    st.write(f"### 🔢 Predicted Number : {pred_label}")
