# AyuSethu ML

> FastAPI inference service for medicinal plant species identification.

**Live:** https://ayusethuml.onrender.com

## Pipeline

```
Input Image (300×300)
  → EfficientNetV2B3 Feature Extractor (1,536 features)
  → PCA Dimensionality Reduction (1,024 components)
  → SVM Classifier (92 species)
  → Top-3 Predictions with Confidence Scores
```

## Tech Stack

FastAPI · TensorFlow 2.18 · Scikit-learn 1.6.1 · Pillow · Gunicorn + Uvicorn

## Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Model readiness check |
| POST | `/api/v1/ml/identify` | Species identification (multipart image) |

### Example Response

```json
{
  "success": true,
  "plant": "Eclipta prostrata (Bringaraja)",
  "confidence": 0.6447,
  "top_predictions": [
    { "plant": "Eclipta prostrata (Bringaraja)", "confidence": 0.6447 },
    { "plant": "Ocimum tenuiflorum (Tulsi)", "confidence": 0.2373 },
    { "plant": "Spinacia oleracea (Palak(Spinach))", "confidence": 0.0872 }
  ]
}
```

## Model Artifacts

| File | Size | Description |
|------|------|-------------|
| `feature_extractor.keras` | 51 MB | EfficientNetV2B3 (pretrained) |
| `pca.pkl` | 6 MB | PCA (1,024 components) |
| `svm_model.pkl` | 40 MB | SVM classifier (92 classes) |
| `class_names.json` | 3 KB | Species name mapping |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Test

```bash
curl -X POST http://localhost:8000/api/v1/ml/identify \
  -F "file=@leaf_image.jpg"
```

## Deployment (Render)

- Runtime: Python 3.11
- Start: `gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT --timeout 120`
- Single worker (models are ~100MB in memory)

## License

ISC
