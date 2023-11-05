from flask import Flask, request, render_template, send_file
import os
import cv2
from PIL import Image
import numpy as np
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import concatenate, BatchNormalization, Dropout, Lambda
from keras import backend as K
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import segmentation_models as sm
import tensorflow as tf
import numpy as np
import io
import base64
import matplotlib.cm as cm

# Initialize Flask app
app = Flask(__name__)

# Load the model

def jaccard_coef(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    final_coef_value = (intersection + 1.0) / (
        K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0
    )
    return final_coef_value

def custom_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + (1 * focal_loss(y_true, y_pred))

metrics = ["accuracy", jaccard_coef]
weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
loaded_model = tf.keras.models.load_model("Modelsarath591.h5", custom_objects={"custom_loss": custom_loss, "jaccard_coef": jaccard_coef})


def preprocess1(image):
    image_patch_size = 256
    size_x = (image.shape[1] // image_patch_size) * image_patch_size
    size_y = (image.shape[0] // image_patch_size) * image_patch_size
    image = Image.fromarray(image)
    image = image.crop((0, 0, size_x, size_y))
    image = np.array(image)
    image_patches = patchify(
        image, (image_patch_size, image_patch_size, 3), step=image_patch_size
    )
    minmaxscaler = MinMaxScaler()
    image_x = image_patches[0, 0, :, :]
    image_y = minmaxscaler.fit_transform(
        image_x.reshape(-1, image_x.shape[-1])
    ).reshape(image_x.shape)
    image_y = image_y[0]
    images = np.array(image_y)
    return images


def prediction(image):
    preprocessed_image = preprocess1(image)
    print(preprocessed_image.min(), preprocessed_image.max())
    plt.imshow(preprocessed_image, cmap="gray")
    plt.show()
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    predicted_mask = loaded_model.predict(preprocessed_image)
    predicted_image = np.argmax(predicted_mask, axis=3)
    predicted_image = predicted_image[0, :, :]
    return predicted_image

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
    # Get the uploaded image file
        uploaded_image = request.files["image"]
        if uploaded_image.filename != "":
            # Read the image using OpenCV
            image = cv2.imdecode(
                np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR
            )
            # Make a prediction
            predicted_image = prediction(image)
            # Debugging code
            print(predicted_image.min(), predicted_image.max())
            plt.imshow(predicted_image, cmap="viridis")
            plt.show()
            normalized_image = (predicted_image - predicted_image.min()) / (predicted_image.max() - predicted_image.min())
            # Apply the 'viridis' colormap (or any other colormap) to the normalized image.
            colored_image = (cm.viridis(normalized_image) * 255).astype(np.uint8)
            # Define the filename to save the colored image.
            result_filename = "predicted_image_colored.png"
            # Save the colored image.
            cv2.imwrite(result_filename, colored_image)
            # Return the saved image as a response
            return send_file(result_filename, mimetype="image/png")

    return render_template("index.html", result=None)
@app.route("/result_image")
def result_image():
    # Serve the result image as a static file
    result_filename = "predicted_image.png"
    return send_file(result_filename, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
