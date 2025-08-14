# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os, io, math, time
import numpy as np
from PIL import Image
import joblib
import tensorflow as tf
import requests

# ---------------------- Config ----------------------
MODELS_DIR = os.getenv("MODELS_DIR", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Override these via env if you ever change locations
HTTP_CNN_URL = os.getenv("HTTP_CNN_URL", "https://lamtonbucket.s3.us-east-2.amazonaws.com/models/cnn_pneumonia_model.h5")
HTTP_RF_URL  = os.getenv("HTTP_RF_URL",  "https://lamtonbucket.s3.us-east-2.amazonaws.com/models/random_forest_model.pkl")
HTTP_SVM_URL = os.getenv("HTTP_SVM_URL", "https://lamtonbucket.s3.us-east-2.amazonaws.com/models/svm_model.pkl")
HTTP_DT_URL  = os.getenv("HTTP_DT_URL",  "https://lamtonbucket.s3.us-east-2.amazonaws.com/models/decision_tree_model.pkl")
HTTP_LE_URL  = os.getenv("HTTP_LE_URL",  "https://lamtonbucket.s3.us-east-2.amazonaws.com/models/label_encoder.pkl")

# ---------------------- App & CORS ----------------------
app = FastAPI(title="Model Inference API (HTTP object URLs)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- HTTP downloader (ETag/Last-Modified aware) ----------------------
def _etag_path(local_path: str) -> str:
    return f"{local_path}.etag"

def ensure_local_from_http(url: str, local_path: str) -> str:
    """Download via HTTP(S) if missing or if ETag/Last-Modified changed."""
    if not url:
        raise HTTPException(500, f"HTTP URL not configured for {local_path}")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # HEAD to check version
    try:
        h = requests.head(url, allow_redirects=True, timeout=20)
        h.raise_for_status()
    except Exception as e:
        raise HTTPException(500, f"HEAD {url} failed: {e}")
    etag = (h.headers.get("ETag") or "").strip('"')
    last_mod = h.headers.get("Last-Modified") or ""
    content_len = h.headers.get("Content-Length") or ""
    version = etag or last_mod or content_len

    etag_file = _etag_path(local_path)
    local_version = ""
    if os.path.exists(local_path) and os.path.exists(etag_file):
        try:
            with open(etag_file, "r") as f:
                local_version = f.read().strip()
        except Exception:
            local_version = ""

    # Download if file missing or version changed
    if (not os.path.exists(local_path)) or (version and version != local_version):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            raise HTTPException(500, f"GET {url} failed: {e}")
        with open(etag_file, "w") as f:
            f.write(version)

    return local_path

# ---------------------- Registry ----------------------
REGISTRY: Dict[str, Dict[str, Any]] = {
    "cnn": {
        "name": "CNN Pneumonia",
        "framework": "keras",
        "type": "image",
        "http_url": HTTP_CNN_URL,
        "path": os.path.join(MODELS_DIR, "cnn_pneumonia_model.h5"),
        "class_names": ["NORMAL", "PNEUMONIA"],  # keep training order
        "_obj": None,
    },
    "rf": {
        "name": "Random Forest (image)",
        "framework": "sklearn",
        "type": "image",  # set to "tabular" if this RF is for numeric features
        "http_url": HTTP_RF_URL,
        "path": os.path.join(MODELS_DIR, "random_forest_model.pkl"),
        "label_encoder_http_url": HTTP_LE_URL,
        "label_encoder_path": os.path.join(MODELS_DIR, "label_encoder.pkl"),
        "normalize_default": False,  # True if trained on pixels/255
        "_obj": None,
        "_le": None,
    },
    "svm": {
        "name": "SVM (image)",
        "framework": "sklearn",
        "type": "image",
        "http_url": HTTP_SVM_URL,
        "path": os.path.join(MODELS_DIR, "svm_model.pkl"),
        "label_encoder_http_url": HTTP_LE_URL,
        "label_encoder_path": os.path.join(MODELS_DIR, "label_encoder.pkl"),
        "normalize_default": False,
        "_obj": None,
        "_le": None,
    },
    "dt": {
        "name": "Decision Tree (image)",
        "framework": "sklearn",
        "type": "image",
        "http_url": HTTP_DT_URL,
        "path": os.path.join(MODELS_DIR, "decision_tree_model.pkl"),
        "label_encoder_http_url": HTTP_LE_URL,
        "label_encoder_path": os.path.join(MODELS_DIR, "label_encoder.pkl"),
        "normalize_default": False,
        "_obj": None,
        "_le": None,
    },
}

# ----- add this helper -----
def _load_keras_compat(path: str):
    """
    Load Keras models saved with older tf.keras configs on newer Keras.
    Works on Keras 3 (needs safe_mode=False) and Keras 2.x (no safe_mode arg).
    """
    try:
        return tf.keras.models.load_model(path, compile=False, safe_mode=False)
    except TypeError:
        # Older tf.keras that doesn't support safe_mode
        return tf.keras.models.load_model(path, compile=False)


def _load_entry(mid: str) -> Dict[str, Any]:
    if mid not in REGISTRY:
        raise HTTPException(404, f"Unknown model id: {mid}")
    e = REGISTRY[mid]

    # Ensure local files exist (download/update over HTTP)
    if "http_url" in e and e["http_url"]:
        ensure_local_from_http(e["http_url"], e["path"])
    if "label_encoder_http_url" in e and e["label_encoder_http_url"]:
        ensure_local_from_http(e["label_encoder_http_url"], e["label_encoder_path"])

    # Lazy load into memory
    if e["_obj"] is None:
        if e["framework"] == "keras":
            e["_obj"] = _load_keras_compat(e["path"]) 
        elif e["framework"] == "sklearn":
            e["_obj"] = joblib.load(e["path"])
            le_path = e.get("label_encoder_path")
            if le_path and os.path.exists(le_path):
                try:
                    e["_le"] = joblib.load(le_path)
                except Exception:
                    e["_le"] = None
        else:
            raise HTTPException(500, f"Unsupported framework: {e['framework']}")
    return e

# ---------------------- Preprocessing utils ----------------------
def preprocess_for_keras(img_bytes: bytes, model: tf.keras.Model) -> np.ndarray:
    """Resize/color-match to the model's input shape, normalize to 0..1."""
    ishape = model.input_shape  # (None, H, W, C)
    if not (isinstance(ishape, tuple) and len(ishape) == 4):
        raise HTTPException(500, f"Unexpected Keras input shape: {ishape}")
    H, W, C = ishape[1], ishape[2], ishape[3]
    if None in (H, W, C):  # fallback if saved with dynamic dims
        H, W, C = 150, 150, 1

    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert("RGB" if C == 3 else "L")
    img = img.resize((W, H))
    arr = np.array(img).astype("float32") / 255.0
    if C == 1 and arr.ndim == 2:
        arr = np.expand_dims(arr, -1)
    arr = np.expand_dims(arr, 0)  # (1, H, W, C)
    return arr

def _closest_hw(n: int) -> tuple[int, int]:
    r = int(math.sqrt(n))
    for h in range(r, 0, -1):
        if n % h == 0:
            return h, n // h
    return 1, n

def infer_hw_c(n_features: int) -> tuple[int, int, int]:
    """
    Best-effort guess of H, W, C from a flat vector length:
      - perfect square -> grayscale square
      - divisible by 3 with square (n/3) -> RGB square
      - else near-square grayscale (prefer RGB if more square)
    """
    s = int(math.isqrt(n_features))
    if s * s == n_features:
        return s, s, 1
    if n_features % 3 == 0:
        s3 = int(math.isqrt(n_features // 3))
        if s3 * s3 == n_features // 3:
            return s3, s3, 3
    h, w = _closest_hw(n_features)
    if n_features % 3 == 0:
        h3, w3 = _closest_hw(n_features // 3)
        if abs(h3 - w3) < abs(h - w):
            return h3, w3, 3
    return h, w, 1

def preprocess_flat_pixels(img_bytes: bytes, H: int, W: int, C: int, normalize: bool) -> np.ndarray:
    """Convert an image to a flat pixel vector (1, H*W*C), optionally /255."""
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert("RGB" if C == 3 else "L").resize((W, H))
    arr = np.array(img).astype("float32")
    if normalize:
        arr = arr / 255.0
    if C == 1 and arr.ndim == 3:
        arr = arr[..., 0]
    if C == 3 and arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr.flatten().reshape(1, -1)

# ---------------------- Endpoints ----------------------
@app.get("/healthz")
def healthz():
    return {"ok": True, "mode": "http", "models_dir": MODELS_DIR}

@app.get("/models")
def list_models():
    """For your Next.js dropdown."""
    return [
        {"id": mid, "name": e["name"], "type": e["type"], "framework": e["framework"]}
        for mid, e in REGISTRY.items()
    ]

class TabularInput(BaseModel):
    values: List[float]

@app.post("/predict_tabular")
def predict_tabular(
    body: TabularInput,
    model: str = Query(..., description="Model id (type must be 'tabular')"),
):
    e = _load_entry(model)
    if e["type"] != "tabular":
        raise HTTPException(400, f"Model '{model}' is not a tabular model.")
    m = e["_obj"]
    X = np.array(body.values, dtype="float32").reshape(1, -1)
    y = m.predict(X)[0]
    label = y
    if e.get("_le") is not None:
        try:
            label = e["_le"].inverse_transform([y])[0]
        except Exception:
            pass
    resp = {"model": model, "prediction": label}
    if hasattr(m, "predict_proba"):
        try:
            resp["probs"] = m.predict_proba(X)[0].astype(float).tolist()
        except Exception:
            pass
    return resp

@app.get("/versions")
def versions():
    import tensorflow as tf, keras, h5py, numpy
    return {
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "h5py": h5py.__version__,
        "numpy": numpy.__version__,
    }


@app.post("/predict_image")
async def predict_image(
    file: UploadFile = File(...),
    model: str = Query("cnn", description="Model id: cnn | rf | svm | dt"),
    normalize: Optional[bool] = Query(None, description="For sklearn image models: divide pixels by 255"),
    force_h: Optional[int] = Query(None, description="Override inferred H (sklearn only)"),
    force_w: Optional[int] = Query(None, description="Override inferred W (sklearn only)"),
    force_c: Optional[int] = Query(None, description="Override channels 1 or 3 (sklearn only)"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image.")
    e = _load_entry(model)

    # ----- Keras CNN path -----
    if e["framework"] == "keras":
        m: tf.keras.Model = e["_obj"]
        x = preprocess_for_keras(await file.read(), m)
        t0 = time.perf_counter()
        preds = m.predict(x)
        ms = (time.perf_counter() - t0) * 1000

        class_names = e["class_names"]
        # (1,1) sigmoid vs (1,C) softmax
        if preds.ndim == 2 and preds.shape[1] == 1:
            score = float(preds[0, 0])
            idx = int(score >= 0.5)
            conf = score if idx == 1 else 1.0 - score
            return {
                "model": model,
                "label": class_names[idx],
                "confidence": round(conf * 100, 2),
                "score": score,
                "inference_ms": round(ms, 2),
            }
        elif preds.ndim == 2 and preds.shape[0] == 1:
            idx = int(np.argmax(preds[0]))
            conf = float(preds[0, idx])
            return {
                "model": model,
                "label": class_names[idx],
                "confidence": round(conf * 100, 2),
                "index": idx,
                "probs": preds[0].astype(float).tolist(),
                "inference_ms": round(ms, 2),
            }
        else:
            raise HTTPException(500, f"Unexpected prediction shape: {preds.shape}")

    # ----- sklearn (image → flat pixels) path -----
    if e["framework"] == "sklearn":
        m = e["_obj"]
        if not hasattr(m, "n_features_in_"):
            raise HTTPException(400, "Model lacks n_features_in_; ensure it was trained in sklearn.")
        n = int(m.n_features_in_)
        # infer H, W, C; allow overrides
        H, W, C = infer_hw_c(n)
        if force_h and force_w:
            H, W = int(force_h), int(force_w)
        if force_c in (1, 3):
            C = int(force_c)
        if H * W * C != n:
            if n % (H * C) == 0:
                W = n // (H * C)
            else:
                raise HTTPException(400, f"Model expects {n} features; got {H}x{W}x{C}={H*W*C}. Provide force_h/force_w/force_c.")
        # default normalization (overridable via query)
        if normalize is None:
            normalize = bool(e.get("normalize_default", False))

        x = preprocess_flat_pixels(await file.read(), H, W, C, normalize)

        t0 = time.perf_counter()
        y = m.predict(x)[0]
        ms = (time.perf_counter() - t0) * 1000

        label = y
        if e.get("_le") is not None:
            try:
                label = e["_le"].inverse_transform([y])[0]
            except Exception:
                pass

        resp = {
            "model": model,
            "inferred_shape": {"H": H, "W": W, "C": C, "n_features": n},
            "prediction": label,
            "inference_ms": round(ms, 2),
        }
        if hasattr(m, "predict_proba"):
            try:
                resp["probs"] = m.predict_proba(x)[0].astype(float).tolist()
            except Exception:
                pass
        return resp

    raise HTTPException(500, f"Unsupported framework for model '{model}'.")
