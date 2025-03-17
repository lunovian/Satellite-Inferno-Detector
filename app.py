import os
import io
import base64
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
from demo import YOLOEnsemble
from werkzeug.utils import secure_filename
import threading
import concurrent.futures
import traceback

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max upload
app.config["MODELS_DIR"] = "models"
app.config["MAX_WORKERS"] = 4  # For parallel processing

# Create upload folder if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize the YOLO ensemble
ensemble = None

# Keep track of available models
available_models = []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


def process_image(filepath, filename, ensemble, models_to_use):
    """Process a single image and return the detection results"""
    try:
        # Get image for display
        img = cv2.imread(filepath)
        if img is None:
            return {"filename": filename, "error": "Could not read image file"}

        # Get predictions
        predictions = ensemble.predict(filepath, visualize=False)

        # Draw predictions on the image
        img_result = ensemble._draw_predictions(img, predictions)

        # Convert the result image to base64 for display
        _, buffer = cv2.imencode(".jpg", img_result)
        img_str = base64.b64encode(buffer).decode("utf-8")

        # Prepare prediction details for display
        pred_details = []
        for pred in predictions:
            model_idx = pred["model_idx"]
            if model_idx < len(models_to_use):
                model_name = os.path.basename(models_to_use[model_idx])
            else:
                model_name = f"Model {model_idx + 1}"

            pred_details.append(
                {
                    "class_id": int(pred["cls_id"]),
                    "confidence": float(pred["conf"]),
                    "model": int(model_idx + 1),
                    "model_name": model_name,
                    "box": pred["box"].tolist(),
                }
            )

        return {
            "filename": filename,
            "image": img_str,
            "predictions": pred_details,
            "count": len(predictions),
        }
    except AttributeError as e:
        if "'Conv' object has no attribute 'bn'" in str(e):
            return {
                "filename": filename,
                "error": "Model compatibility error: 'Conv' object has no attribute 'bn'. This usually happens with YOLOv10 models running on older Ultralytics versions. Please update to Ultralytics 8.1.0 or later.",
            }
        else:
            return {"filename": filename, "error": f"AttributeError: {str(e)}"}
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error processing {filename}: {error_trace}")
        return {"filename": filename, "error": str(e)}


@app.route("/predict", methods=["POST"])
def predict():
    if "files[]" not in request.files:
        return jsonify({"error": "No files part"})

    files = request.files.getlist("files[]")
    if not files or files[0].filename == "":
        return jsonify({"error": "No selected files"})

    # Get parameters
    conf_thres = float(request.form.get("conf_threshold", 0.3))
    iou_thres = float(request.form.get("iou_threshold", 0.5))

    # Get selected models
    selected_models = request.form.getlist("selected_models[]")
    if not selected_models:
        return jsonify({"error": "No models selected"})

    # Map selected model names to indices in the available_models list
    models_to_use = []
    for model_name in selected_models:
        if model_name in available_models:
            model_path = os.path.join(app.config["MODELS_DIR"], model_name)
            models_to_use.append(model_path)

    if not models_to_use:
        return jsonify({"error": "None of the selected models are available"})

    # Initialize YOLO ensemble with selected models
    global ensemble
    ensemble = YOLOEnsemble(
        models_dir=None,  # We'll pass specific model paths instead
        conf_thres=conf_thres,
        iou_thres=iou_thres,
    )

    # Override the models list with just the selected models
    ensemble.models = []
    compatible_models = []
    incompatible_models = []

    for model_path in models_to_use:
        model_name = os.path.basename(model_path)
        print(f"Loading {model_name}...")
        try:
            from ultralytics import YOLO

            # Check if it's likely a YOLOv10 model based on filename
            is_likely_v10 = (
                "v10" in model_name.lower() or "yolov10" in model_name.lower()
            )

            # Add warning for potential v10 models
            if is_likely_v10:
                print(
                    f"Warning: {model_name} appears to be a YOLOv10 model which may require Ultralytics 8.1.0+"
                )

            model = YOLO(model_path)
            ensemble.models.append(model)
            compatible_models.append(model_path)
        except Exception as e:
            error_msg = str(e)
            print(f"Error loading model {model_name}: {error_msg}")

            # Provide more specific error for v10 models
            if "'Conv' object has no attribute 'bn'" in error_msg:
                if "v10" in model_name.lower() or "yolov10" in model_name.lower():
                    error_msg = f"YOLOv10 model compatibility error: {error_msg}. YOLOv10 requires Ultralytics 8.1.0+"

            incompatible_models.append((model_name, error_msg))

    if len(ensemble.models) == 0:
        # No compatible models were loaded
        error_details = "\n".join(
            [f"{name}: {error}" for name, error in incompatible_models]
        )
        return jsonify(
            {
                "error": f"Unable to load any selected models. Please check model compatibility with your version of Ultralytics library.\n\nDetails:\n{error_details}"
            }
        )

    # Save the uploaded files
    valid_files = []
    for file in files:
        if file.filename == "":
            continue

        # Check if this is an image file
        filename_lower = file.filename.lower()
        if not any(
            filename_lower.endswith(ext)
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        ):
            continue

        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        valid_files.append((filepath, filename))

    if not valid_files:
        return jsonify({"error": "No valid image files were uploaded"})

    # Process each file
    results = []

    # Use thread pool for parallel processing if there are multiple images
    if len(valid_files) > 1:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=app.config["MAX_WORKERS"]
        ) as executor:
            future_to_file = {
                executor.submit(
                    process_image, filepath, filename, ensemble, models_to_use
                ): filename
                for filepath, filename in valid_files
            }

            for future in concurrent.futures.as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"filename": filename, "error": str(e)})
    else:
        # Process single file directly
        filepath, filename = valid_files[0]
        results.append(process_image(filepath, filename, ensemble, models_to_use))

    return jsonify(
        {
            "success": True,
            "results": results,
            "total_files": len(results),
            "models_used": [os.path.basename(m) for m in models_to_use],
        }
    )


@app.route("/model_info")
def model_info():
    # Get list of available models
    global available_models
    available_models = []
    if os.path.exists(app.config["MODELS_DIR"]):
        available_models = [
            f for f in os.listdir(app.config["MODELS_DIR"]) if f.endswith(".pt")
        ]

    return jsonify(
        {"available_models": available_models, "models_count": len(available_models)}
    )


@app.route("/upload_folder", methods=["POST"])
def upload_folder():
    return jsonify({"success": True, "message": "Folder upload is not yet implemented"})


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large. Maximum file size is 100MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
