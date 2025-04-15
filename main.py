import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the MNIST model
model = load_model("models/mnist_model.keras")

# Initialize drawing canvas
canvas_size = 280
canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
drawing = False
ix, iy = -1, -1

def preprocess_image(image):
    """Resize and normalize image for prediction"""
    resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0  # normalize to [0, 1]
    reshaped = normalized.reshape(1, 28, 28, 1)  # shape for model
    return reshaped

def predict_digit(image):
    """Preprocess and predict the digit"""
    processed = preprocess_image(image)
    prediction = model.predict(processed, verbose=0)
    predicted_digit = np.argmax(prediction)
    return predicted_digit

def show_prediction(image, digit):
    """Display the resized image and prediction"""
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted Digit: {digit}")
    plt.axis('off')
    plt.show()

# Drawing handler
def draw(event, x, y, flags, param):
    global drawing, ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(canvas, (ix, iy), (x, y), 255, thickness=20)
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Predict after finishing drawing
        resized = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
        digit = predict_digit(canvas)
        show_prediction(resized, digit)

# Setup OpenCV window and mouse callback
cv2.namedWindow("Draw Digit (Press 'c' to clear, ESC to exit)")
cv2.setMouseCallback("Draw Digit (Press 'c' to clear, ESC to exit)", draw)

# Run loop
while True:
    cv2.imshow("Draw Digit (Press 'c' to clear, ESC to exit)", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Clear canvas
        canvas[:] = 0
    elif key == 27:  # ESC to exit
        break

cv2.destroyAllWindows()