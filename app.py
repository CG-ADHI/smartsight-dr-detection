from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

# ===============================
# FOLDERS
# ===============================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# LOAD MODEL
# ===============================
model = load_model("model/model.h5")

class_names = ["Mild", "Moderate", "No_DR", "Proliferative", "Severe"]

# ===============================
# RISK LEVEL MAPPING
# ===============================
risk_mapping = {
    "No_DR": "Low Risk – Routine eye check recommended",
    "Mild": "Low Risk – Periodic monitoring advised",
    "Moderate": "Medium Risk – Ophthalmologist consultation recommended",
    "Severe": "High Risk – Urgent referral to eye specialist required",
    "Proliferative": "Critical Risk – Immediate specialist consultation required"
}

# ===============================
# CLASS-SPECIFIC EXPLANATIONS
# ===============================
explanation_mapping = {
    "No_DR": "The model observes normal retinal structure with no significant lesions or vascular abnormalities.",
    "Mild": "The prediction is influenced by early signs such as microaneurysms and minor vascular changes.",
    "Moderate": "The model identifies hemorrhages and hard exudates indicating moderate retinal damage.",
    "Severe": "Severe retinal abnormalities including extensive hemorrhages and vascular distortion are detected.",
    "Proliferative": "The model detects abnormal new blood vessel growth, indicating advanced diabetic retinopathy."
}

# ===============================
# BLUR DETECTION
# ===============================
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

# ===============================
# ROUTE
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    risk = None
    explanation = None
    warning = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        img = cv2.imread(image_path)

        if img is None:
            warning = "Unable to read image. Please upload a valid image."
            return render_template("index.html", warning=warning)

        # Blur check
        if is_blurry(img):
            warning = "Image is blurry. Please upload a clearer fundus image."
            return render_template("index.html", warning=warning)

        # Preprocess
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (240, 240))
        img_array = np.expand_dims(img, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        preds = model.predict(img_array)[0]
        idx = np.argmax(preds)

        prediction = class_names[idx]
        confidence = round(preds[idx] * 100, 2)
        risk = risk_mapping[prediction]
        explanation = explanation_mapping[prediction]

        if confidence < 30:
            warning = "Low confidence prediction – Clinical review recommended"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        risk=risk,
        explanation=explanation,
        warning=warning,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
