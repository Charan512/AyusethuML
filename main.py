"""
AyuSethu — FastAPI ML Inference Service
Pipeline: EfficientNetV2B3 Feature Extractor → PCA → SVM (raw predict_proba)
"""

import json
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import joblib
import tensorflow as tf

# ── Global model references ───────────────────────────
feature_extractor = None
pca = None
svm_model = None
class_names = None

MODEL_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all model artifacts into memory at startup."""
    global feature_extractor, pca, svm_model, class_names

    print("🔄 Loading model artifacts...")

    # 1. Load EfficientNetV2B3 feature extractor
    feature_extractor = tf.keras.models.load_model(
        str(MODEL_DIR / "feature_extractor.keras")
    )
    print(f"   ✅ Feature extractor loaded — output shape: {feature_extractor.output_shape}")

    # 2. Load PCA model
    pca = joblib.load(str(MODEL_DIR / "pca.pkl"))
    print(f"   ✅ PCA loaded — n_components: {pca.n_components_}")

    # 3. Load SVM classifier
    svm_model = joblib.load(str(MODEL_DIR / "svm_model.pkl"))
    print(f"   ✅ SVM loaded — n_classes: {len(svm_model.classes_)}")

    # 4. Load class names mapping
    with open(MODEL_DIR / "class_names.json", "r") as f:
        class_names = json.load(f)
    print(f"   ✅ Class names loaded — {len(class_names)} species")

    print("🚀 All models loaded successfully!")
    yield
    print("🛑 Shutting down ML service...")


# ── App init ──────────────────────────────────────────
app = FastAPI(
    title="AyuSethu ML Inference Service",
    description="Botanical species identification via EfficientNetV2B3 + PCA + SVM",
    version="1.0.0",
    lifespan=lifespan,
)

import os

# In production, set CORS_ORIGINS=https://your-api.onrender.com,https://your-frontend.onrender.com
cors_origins_env = os.environ.get("CORS_ORIGINS", "")
allowed_origins = [s.strip() for s in cors_origins_env.split(",") if s.strip()] if cors_origins_env else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────
@app.get("/health")
async def health_check():
    models_loaded = all([
        feature_extractor is not None,
        pca is not None,
        svm_model is not None,
        class_names is not None,
    ])
    return {
        "success": True,
        "status": "healthy" if models_loaded else "models_not_loaded",
        "models": {
            "feature_extractor": feature_extractor is not None,
            "pca": pca is not None,
            "svm_model": svm_model is not None,
            "class_names": class_names is not None,
        },
        "species_count": len(class_names) if class_names else 0,
    }


# ── Inference endpoint ────────────────────────────────
@app.post("/api/v1/ml/identify")
async def identify_plant(file: UploadFile = File(...)):
    """
    Accepts a leaf image, runs the full inference pipeline,
    and returns the top prediction + top 3 alternatives with raw confidence scores.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image (JPEG, PNG, etc.)",
        )

    try:
        # ── 1. Read and preprocess image ──────────────
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((300, 300))

        # Convert to array and apply EfficientNetV2-specific preprocessing
        # IMPORTANT: Do NOT manually scale by 1/255 — preprocess_input handles it
        img_array = np.array(image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 300, 300, 3)
        img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

        # ── 2. Feature extraction ─────────────────────
        x = img_array
        for layer in feature_extractor.layers:
            x = layer(x, training=False)
        features = x.numpy() if hasattr(x, "numpy") else x
        # Flatten to 2D: (1, feature_count) for PCA input
        features = np.array(features).reshape(1, -1)

        # ── 3. PCA dimensionality reduction ───────────
        pca_features = pca.transform(features)

        # ── 4. SVM classification ─────────────────────
        # Handle SVMs trained with or without probability=True
        # STRICT: No manual scaling — use raw scores or built-in predict_proba
        if hasattr(svm_model, 'predict_proba'):
            probabilities = svm_model.predict_proba(pca_features)[0]
        else:
            # Fallback: use decision_function + scipy softmax for confidence scores
            from scipy.special import softmax
            decision_scores = svm_model.decision_function(pca_features)[0]
            probabilities = softmax(decision_scores)

        # ── 5. Build response ─────────────────────────
        top_indices = np.argsort(probabilities)[::-1]

        top_prediction = {
            "plant": class_names[top_indices[0]],
            "confidence": float(probabilities[top_indices[0]]),
        }

        top_predictions = [
            {
                "plant": class_names[idx],
                "confidence": float(probabilities[idx]),
            }
            for idx in top_indices[:3]
        ]

        return {
            "success": True,
            "plant": top_prediction["plant"],
            "confidence": top_prediction["confidence"],
            "top_predictions": top_predictions,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )
