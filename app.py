import os
import sys
import io
import json
import uuid
import logging
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms, models

# ============================================================
# AUTHENTICAI - COMBINED DETECTOR
#
# Pipelines:
#   1. Face Morphing  -> retrained EfficientNet-B0 + Meso-4 + fusion
#                        head (5-fold ensemble of fold*_bundle.pth)
#   2. AI Generated   -> EfficientNet-B1 5-fold ensemble
#   3. Audio Deepfake -> AudioDeepfakeModel
#   4. Video          -> per-frame fusion of (1) + (2), weighted
#                        toward the AI-generated ensemble, then
#                        soft-voted across frames
#   5. Camera         -> image pipeline selected in the UI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
AI_CV_DIR = BASE_DIR / "cv_results_efficientnet_b1_5fold"
UPLOAD_DIR = BASE_DIR / "uploads"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.model import AudioDeepfakeModel

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AuthenticAI")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", DEVICE)

# ============================================================
# MODEL PATHS / SETTINGS
# ============================================================

# ---- Face morphing (retrained, from 03_retrain_meso_fusion.py) ----
# Put fold0_bundle.pth ... fold4_bundle.pth (and best_model.json)
# from DATA/models/cv_v2 / DATA/results_v2 into this folder.
FACE_BUNDLE_DIR = MODELS_DIR / "face_cv_v2"
FACE_BEST_JSON = FACE_BUNDLE_DIR / "best_model.json"

# Which output of the bundle decides the verdict:
#   "auto"   -> read best_variant from best_model.json (fallback: fusion)
#   "fusion" -> fusion head
#   "eff"    -> EfficientNet-B0 only
#   "meso"   -> Meso-4 only
#   "avg"    -> mean of eff and meso
FACE_VARIANT = "auto"
FACE_VARIANTS = ("eff", "meso", "avg", "fusion")

# The models were trained on cropped face frames, so face cropping is
# forced ON for the face-morphing model regardless of the UI checkbox.
FACE_FORCE_CROP = True

FACE_SIZE = 224
FACE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
FACE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# ---- Audio ----
AUDIO_PATH = MODELS_DIR / "best_audio_deepfake_model.pth"

# ---- AI-generated image ensemble ----
# best_model-v3.pt stays in models/ but is NOT used here.
AI_MODEL_NAME = "efficientnet_b1"
AI_IMAGE_SIZE = 240
AI_NUM_FOLDS = 5
AI_DEFAULT_CLASSES = ["Real", "Fake"]

# ---- video fusion weights (should sum to 1.0) ----
VIDEO_AI_GENERATED_WEIGHT = 0.65
VIDEO_FACE_MORPH_WEIGHT = 0.35

AI_TRANSFORM = transforms.Compose([
    transforms.Resize((AI_IMAGE_SIZE, AI_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ============================================================
# FACE DETECTION / PREPROCESSING
#
# Needs opencv-contrib-python (not plain opencv-python) for
# cv2.CascadeClassifier on OpenCV 5:
#     pip uninstall opencv-python -y
#     pip install opencv-contrib-python
# ============================================================

BUNDLED_CASCADE_PATH = MODELS_DIR / "haarcascade_frontalface_default.xml"

_face_detector = None
_face_detector_load_attempted = False


def get_face_detector():
    global _face_detector, _face_detector_load_attempted

    if _face_detector is not None:
        return _face_detector

    if _face_detector_load_attempted:
        return None

    _face_detector_load_attempted = True

    if BUNDLED_CASCADE_PATH.exists():
        cascade_path = str(BUNDLED_CASCADE_PATH)
    else:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        except AttributeError:
            logger.warning("No Haar cascade available. Full frames will be used.")
            return None

    if not os.path.exists(cascade_path):
        logger.warning("Haar Cascade file not found. Full frames will be used.")
        return None

    try:
        classifier = cv2.CascadeClassifier(cascade_path)
        if classifier.empty():
            logger.warning("Haar Cascade failed to load: %s", cascade_path)
            return None
        _face_detector = classifier
        logger.info("Haar Cascade face detector loaded from: %s", cascade_path)
    except AttributeError:
        logger.warning(
            "cv2.CascadeClassifier is unavailable - install opencv-contrib-python: "
            "pip uninstall opencv-python -y && pip install opencv-contrib-python"
        )
        _face_detector = None

    return _face_detector


def detect_and_crop_face(pil_image, margin=0.30):
    cascade = get_face_detector()

    if cascade is None:
        return pil_image

    img_np = np.array(pil_image)
    h, w = img_np.shape[:2]

    if h == 0 or w == 0:
        return pil_image

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    if len(faces) == 0:
        return pil_image

    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    mx = int(fw * margin)
    my = int(fh * margin)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w, x + fw + mx)
    y2 = min(h, y + fh + my)

    if x2 <= x1 or y2 <= y1:
        return pil_image

    return pil_image.crop((x1, y1, x2, y2))


def normalize_image_quality(image):
    """Optional UI 'preprocess' step. NOT used in training; leave off for best accuracy."""
    try:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        max_size = 1080
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")

        img_np = np.array(image)
        img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)

        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])

        img_np = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        image = Image.fromarray(img_np)

        image = ImageEnhance.Sharpness(image).enhance(1.15)
        image = ImageEnhance.Color(image).enhance(0.97)

        return image

    except Exception as exc:
        logger.warning("Image preprocessing warning: %s", exc)
        return image.convert("RGB")


# ============================================================
# FACE MORPHING MODEL DEFINITIONS
# These must match 02_train_cv.py / 03_retrain_meso_fusion.py exactly.
# Label convention in training: 0 = real, 1 = fake.
# ============================================================

class Meso4(nn.Module):
    """Corrected Meso-4 from 03_retrain_meso_fusion.py (224 input -> 16*7*7 features)."""

    def __init__(self, drop=0.3):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8)
        self.c2, self.b2 = nn.Conv2d(8, 8, 5, padding=2), nn.BatchNorm2d(8)
        self.c3, self.b3 = nn.Conv2d(8, 16, 5, padding=2), nn.BatchNorm2d(16)
        self.c4, self.b4 = nn.Conv2d(16, 16, 5, padding=2), nn.BatchNorm2d(16)
        self.drop = nn.Dropout(drop)
        self.fc1, self.fc2 = nn.Linear(16 * 7 * 7, 16), nn.Linear(16, 2)

    def feats(self, x):
        lr = lambda t: F.leaky_relu(t, 0.1)
        x = F.max_pool2d(lr(self.b1(self.c1(x))), 2)
        x = F.max_pool2d(lr(self.b2(self.c2(x))), 2)
        x = F.max_pool2d(lr(self.b3(self.c3(x))), 2)
        x = F.max_pool2d(lr(self.b4(self.c4(x))), 4)
        return lr(self.fc1(self.drop(torch.flatten(x, 1))))

    def forward(self, x):
        return self.fc2(self.drop(self.feats(x)))


class FusionHead(nn.Module):
    """MLP on standardised [EfficientNet 1280 | Meso-4 16] features."""

    def __init__(self, mu, sd):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.net = nn.Sequential(
            nn.Linear(1296, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, f):
        return self.net((f - self.mu) / self.sd)


def build_face_eff():
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(1280, 2)
    return m


class FaceBundle:
    """One fold: EfficientNet-B0 + Meso-4 + fusion head, loaded from fold*_bundle.pth."""

    def __init__(self, path: Path):
        ck = torch.load(path, map_location=DEVICE, weights_only=False)

        self.name = path.name

        self.eff = build_face_eff()
        self.eff.load_state_dict(ck["eff"])
        self.eff.to(DEVICE).eval()

        self.meso = Meso4()
        self.meso.load_state_dict(ck["meso"])
        self.meso.to(DEVICE).eval()

        self.head = FusionHead(torch.zeros(1296), torch.ones(1296))
        self.head.load_state_dict(ck["head"])
        self.head.to(DEVICE).eval()

    @torch.inference_mode()
    def fake_probs(self, x):
        """Returns P(fake) for each variant."""
        fe = torch.flatten(self.eff.avgpool(self.eff.features(x)), 1)
        le = self.eff.classifier(fe)

        fm = self.meso.feats(x)
        lm = self.meso.fc2(fm)

        lf = self.head(torch.cat([fe, fm], dim=1))

        pe = float(torch.softmax(le, 1)[0, 1].cpu())
        pm = float(torch.softmax(lm, 1)[0, 1].cpu())
        pf = float(torch.softmax(lf, 1)[0, 1].cpu())

        return {
            "eff": pe,
            "meso": pm,
            "avg": (pe + pm) / 2.0,
            "fusion": pf
        }


# ============================================================
# LOAD FACE MORPHING + AUDIO MODELS
# ============================================================

face_bundles = []
face_variant = "fusion"
audio_model = None


def resolve_face_variant():
    if FACE_VARIANT in FACE_VARIANTS:
        return FACE_VARIANT

    if FACE_BEST_JSON.exists():
        try:
            with open(FACE_BEST_JSON, "r", encoding="utf-8") as f:
                best = json.load(f).get("best_variant")
            if best in FACE_VARIANTS:
                return best
            logger.warning("Unknown best_variant '%s' in %s", best, FACE_BEST_JSON)
        except Exception as exc:
            logger.warning("Could not read %s: %s", FACE_BEST_JSON, exc)

    return "fusion"


def load_face_models():
    global face_bundles, face_variant

    face_bundles = []

    if not FACE_BUNDLE_DIR.exists():
        raise FileNotFoundError(
            f"Face model folder not found: {FACE_BUNDLE_DIR}. "
            f"Copy fold*_bundle.pth from DATA/models/cv_v2 into it."
        )

    paths = sorted(FACE_BUNDLE_DIR.glob("fold*_bundle.pth"))

    if not paths:
        fallback = FACE_BUNDLE_DIR / "best_model_bundle.pth"
        if fallback.exists():
            paths = [fallback]
            logger.warning("No fold bundles found; using single best_model_bundle.pth.")

    if not paths:
        raise FileNotFoundError(
            f"No fold*_bundle.pth or best_model_bundle.pth in {FACE_BUNDLE_DIR}"
        )

    for path in paths:
        try:
            face_bundles.append(FaceBundle(path))
            logger.info("Face bundle loaded: %s", path.name)
        except Exception as exc:
            logger.exception("Face bundle %s failed to load: %s", path.name, exc)

    if not face_bundles:
        raise RuntimeError("No face-morphing bundles could be loaded.")

    face_variant = resolve_face_variant()
    logger.info(
        "Face morphing ready: %d bundle(s), decision variant = %s",
        len(face_bundles),
        face_variant
    )


def load_audio_model():
    global audio_model

    if not AUDIO_PATH.exists():
        logger.warning("Audio model file not found: %s", AUDIO_PATH)
        return

    try:
        audio_model = AudioDeepfakeModel()

        checkpoint = torch.load(
            AUDIO_PATH,
            map_location=DEVICE,
            weights_only=False
        )

        try:
            audio_model.load_state_dict(checkpoint, strict=True)
        except RuntimeError:
            audio_model.load_state_dict(checkpoint, strict=False)

        audio_model.to(DEVICE).eval()
        logger.info("Audio deepfake model loaded.")

    except Exception as exc:
        audio_model = None
        logger.exception("Audio model failed to load: %s", exc)


# ============================================================
# AI-GENERATED IMAGE MODEL (unchanged)
# ============================================================

ai_ensemble_models = []
ai_ensemble_fold_ids = []
ai_class_names = AI_DEFAULT_CLASSES.copy()
ai_fake_index = 1
ai_real_index = 0


def load_ai_checkpoint(model_path: Path):
    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]

        saved_classes = checkpoint.get("class_names")
        if saved_classes and len(saved_classes) == 2:
            local_classes = list(saved_classes)
        else:
            local_classes = AI_DEFAULT_CLASSES.copy()

        saved_model_name = checkpoint.get("model_name", AI_MODEL_NAME)
        if saved_model_name != AI_MODEL_NAME:
            raise RuntimeError(
                f"{model_path.name} was trained with "
                f"'{saved_model_name}', expected '{AI_MODEL_NAME}'."
            )

        saved_image_size = checkpoint.get("image_size", AI_IMAGE_SIZE)
        if saved_image_size != AI_IMAGE_SIZE:
            logger.warning(
                "%s image size is %s; app uses %s.",
                model_path.name,
                saved_image_size,
                AI_IMAGE_SIZE
            )

    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        local_classes = AI_DEFAULT_CLASSES.copy()

    else:
        state_dict = checkpoint
        local_classes = AI_DEFAULT_CLASSES.copy()

    fold_model = timm.create_model(
        AI_MODEL_NAME,
        pretrained=False,
        num_classes=2
    )

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned_state_dict[key] = value

    missing, unexpected = fold_model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    if missing:
        logger.warning("Missing keys in %s: %s", model_path.name, missing)

    if unexpected:
        logger.warning("Unexpected keys in %s: %s", model_path.name, unexpected)

    fold_model.to(DEVICE).eval()

    return fold_model, local_classes


def load_ai_models():
    global ai_ensemble_models
    global ai_ensemble_fold_ids
    global ai_class_names
    global ai_fake_index
    global ai_real_index

    ai_ensemble_models = []
    ai_ensemble_fold_ids = []

    if not AI_CV_DIR.exists():
        raise FileNotFoundError(
            f"AI-generated model folder not found: {AI_CV_DIR}"
        )

    resolved_classes = None

    for fold in range(1, AI_NUM_FOLDS + 1):
        model_path = AI_CV_DIR / f"fold_{fold}" / "best_model.pth"

        if not model_path.exists():
            logger.warning("AI fold %d missing: %s", fold, model_path)
            continue

        try:
            model, local_classes = load_ai_checkpoint(model_path)

            ai_ensemble_models.append(model)
            ai_ensemble_fold_ids.append(fold)

            if resolved_classes is None:
                resolved_classes = local_classes

            logger.info("AI-generated fold %d loaded.", fold)

        except Exception as exc:
            logger.exception("AI-generated fold %d failed: %s", fold, exc)

    if not ai_ensemble_models:
        raise RuntimeError(
            "No AI-generated EfficientNet-B1 fold models could be loaded."
        )

    ai_class_names = resolved_classes or AI_DEFAULT_CLASSES.copy()

    lower_names = [name.lower() for name in ai_class_names]

    ai_fake_index = lower_names.index("fake") if "fake" in lower_names else 1
    ai_real_index = lower_names.index("real") if "real" in lower_names else 0

    logger.info(
        "AI-generated ensemble loaded: folds=%s classes=%s",
        ai_ensemble_fold_ids,
        ai_class_names
    )


# Load all models when Flask starts.
load_face_models()
load_audio_model()
load_ai_models()


# ============================================================
# FACE MORPHING PREDICTION
# ============================================================

def face_to_tensor(pil_image, flip=False):
    """Same preprocessing as training: RGB, INTER_AREA resize to 224, ImageNet norm."""
    arr = np.array(pil_image.convert("RGB"))

    if arr.shape[:2] != (FACE_SIZE, FACE_SIZE):
        arr = cv2.resize(arr, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)

    if flip:
        arr = arr[:, ::-1]

    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float().div_(255.0)
    t = (t - FACE_MEAN) / FACE_STD

    return t.unsqueeze(0).to(DEVICE)


def predict_face_morphing(
    image,
    threshold=0.5,
    use_tta=False,
    use_face_crop=False,
    use_preprocess=False
):
    """
    raw_score / fake_probability = P(fake), averaged over the fold
    ensemble (and over original + horizontal flip when TTA is on).
    Verdict: Fake when P(fake) >= threshold.
    """
    if not face_bundles:
        raise RuntimeError("Face morphing model is not loaded.")

    image = image.convert("RGB")

    if use_preprocess:
        image = normalize_image_quality(image)

    crop_applied = bool(use_face_crop or FACE_FORCE_CROP)
    if crop_applied:
        image = detect_and_crop_face(image)

    tensors = [face_to_tensor(image)]
    if use_tta:
        tensors.append(face_to_tensor(image, flip=True))

    per_variant = {v: [] for v in FACE_VARIANTS}

    for bundle in face_bundles:
        for tensor in tensors:
            probs = bundle.fake_probs(tensor)
            for v in FACE_VARIANTS:
                per_variant[v].append(probs[v])

    variant_probs = {v: float(np.mean(per_variant[v])) for v in FACE_VARIANTS}

    fake_prob = variant_probs[face_variant]

    prediction = "Fake" if fake_prob >= threshold else "Real"
    confidence = max(fake_prob, 1 - fake_prob) * 100

    return {
        "prediction": prediction,
        "label": "FACE MORPHING" if prediction == "Fake" else "REAL",
        "confidence": round(confidence, 2),
        "raw_score": round(fake_prob, 4),
        "fake_probability": round(fake_prob, 4),
        "pipeline": "Face Morphing",
        "variant": face_variant,
        "ensemble_size": len(face_bundles),
        "variant_probabilities": {
            v: round(p, 4) for v, p in variant_probs.items()
        },
        "tta_enabled": use_tta,
        "face_crop_enabled": crop_applied,
        "preprocess_enabled": use_preprocess
    }


# ============================================================
# AUDIO DEEPFAKE (unchanged)
# ============================================================

def extract_audio_mfcc(audio_path, n_mfcc=40, sr=16000):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=2048,
        hop_length=512
    )

    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)

    features = (
        features - features.mean(axis=1, keepdims=True)
    ) / (
        features.std(axis=1, keepdims=True) + 1e-8
    )

    return features


def predict_audio(audio_path, threshold=0.5):
    if audio_model is None:
        raise RuntimeError("Audio model is not loaded.")

    features = extract_audio_mfcc(audio_path)

    features_resized = cv2.resize(features, (224, 224))

    f_min = features_resized.min()
    f_max = features_resized.max()

    if f_max > f_min:
        features_resized = (features_resized - f_min) / (f_max - f_min)

    tensor = (
        torch.FloatTensor(features_resized)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.inference_mode():
        output = audio_model(tensor)
        probabilities = torch.softmax(output, dim=1)
        raw_score = float(probabilities[0][1].cpu())

    prediction = "Real" if raw_score >= threshold else "Fake"
    confidence = max(raw_score, 1 - raw_score) * 100

    return {
        "prediction": prediction,
        "label": "AUDIO DEEPFAKE" if prediction == "Fake" else "REAL",
        "confidence": round(confidence, 2),
        "raw_score": round(raw_score, 4),
        "pipeline": "Audio Deepfake"
    }


# ============================================================
# VIDEO PREDICTION
#
# Each sampled frame is scored by BOTH specialist pipelines:
#   - the face-morphing model (face-swap / morph artifacts)
#   - the AI-generated EfficientNet-B1 ensemble (fully synthetic)
# Per-frame fake probabilities are blended using
# VIDEO_AI_GENERATED_WEIGHT / VIDEO_FACE_MORPH_WEIGHT, and the
# video verdict is a soft vote (mean fused probability, thresholded).
# ============================================================

def extract_frames(video_path, max_frames=20):
    frames = []

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fast path: seek to evenly spaced frames instead of decoding everything.
    if total > 0:
        indices = np.linspace(0, total - 1, min(max_frames, total)).astype(int)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()

    # Fallback: sequential read (container with unreliable frame count / seeking).
    if not frames:
        cap = cv2.VideoCapture(video_path)
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if count % 10 == 0:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                if len(frames) >= max_frames:
                    break

            count += 1

        cap.release()

    return frames


def _face_morph_fake_probability(frame, threshold, use_tta, use_face_crop, use_preprocess):
    """P(fake) from the face-morphing pipeline (already fake-oriented)."""
    result = predict_face_morphing(
        frame,
        threshold=threshold,
        use_tta=use_tta,
        use_face_crop=use_face_crop,
        use_preprocess=use_preprocess
    )
    return result["raw_score"]


def _ai_generated_fake_probability(frame):
    """P(fake) from the AI-generated ensemble."""
    result = predict_ai_generated(frame)
    return result["raw_score"]


def predict_video(
    video_path,
    threshold=0.5,
    use_tta=False,
    use_face_crop=False,
    use_preprocess=False
):
    frames = extract_frames(video_path)

    if not frames:
        raise RuntimeError("No frames could be extracted from the video.")

    fused_fake_probs = []
    ai_fake_probs = []
    face_fake_probs = []

    for frame in frames:
        try:
            ai_fake_prob = _ai_generated_fake_probability(frame)
        except Exception as exc:
            logger.warning("Video frame AI-generated scoring error: %s", exc)
            continue

        try:
            face_fake_prob = _face_morph_fake_probability(
                frame,
                threshold=0.5,
                use_tta=use_tta,
                use_face_crop=use_face_crop,
                use_preprocess=use_preprocess
            )
        except Exception as exc:
            logger.warning("Video frame face-morphing scoring error: %s", exc)
            continue

        fused = (
            VIDEO_AI_GENERATED_WEIGHT * ai_fake_prob
            + VIDEO_FACE_MORPH_WEIGHT * face_fake_prob
        )

        fused_fake_probs.append(fused)
        ai_fake_probs.append(ai_fake_prob)
        face_fake_probs.append(face_fake_prob)

    if not fused_fake_probs:
        raise RuntimeError("No video frames could be predicted.")

    video_fake_prob = float(np.mean(fused_fake_probs))
    final_prediction = "Fake" if video_fake_prob >= threshold else "Real"
    confidence = max(video_fake_prob, 1 - video_fake_prob) * 100

    fake_frames = sum(1 for p in fused_fake_probs if p >= threshold)

    return {
        "prediction": final_prediction,
        "label": (
            "MANIPULATED VIDEO"
            if final_prediction == "Fake"
            else "REAL VIDEO"
        ),
        "confidence": round(confidence, 2),
        "raw_score": round(video_fake_prob, 4),
        "pipeline": "Video / Combined Analysis",
        "frames_analyzed": len(fused_fake_probs),
        "fake_frames": fake_frames,
        "ai_generated_weight": VIDEO_AI_GENERATED_WEIGHT,
        "face_morph_weight": VIDEO_FACE_MORPH_WEIGHT,
        "ai_generated_avg_fake_prob": round(float(np.mean(ai_fake_probs)), 4),
        "face_morph_avg_fake_prob": round(float(np.mean(face_fake_probs)), 4)
    }


# ============================================================
# AI-GENERATED IMAGE PREDICTION (unchanged)
# ============================================================

def predict_ai_generated(image):
    if not ai_ensemble_models:
        raise RuntimeError("AI-generated model ensemble is not loaded.")

    image = image.convert("RGB")
    tensor = AI_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        summed_probabilities = None

        for model in ai_ensemble_models:
            logits = model(tensor)
            probabilities = F.softmax(logits, dim=1)[0]

            if summed_probabilities is None:
                summed_probabilities = probabilities.clone()
            else:
                summed_probabilities += probabilities

        probabilities = summed_probabilities / len(ai_ensemble_models)

        predicted_index = int(torch.argmax(probabilities).item())

    predicted_class = ai_class_names[predicted_index]

    fake_probability = float(probabilities[ai_fake_index].item() * 100)
    real_probability = float(probabilities[ai_real_index].item() * 100)
    confidence = float(probabilities[predicted_index].item() * 100)

    is_fake = predicted_index == ai_fake_index

    return {
        "prediction": "Fake" if is_fake else "Real",
        "label": "AI GENERATED" if is_fake else "REAL",
        "class_name": predicted_class,
        "confidence": round(confidence, 2),
        "raw_score": round(fake_probability / 100.0, 4),
        "probabilities": {
            "AI GENERATED": round(fake_probability, 2),
            "REAL": round(real_probability, 2)
        },
        "pipeline": "AI-Generated Image",
        "model_name": AI_MODEL_NAME,
        "image_size": AI_IMAGE_SIZE,
        "ensemble_size": len(ai_ensemble_models),
        "folds_used": ai_ensemble_fold_ids
    }


# ============================================================
# ROUTES
# ============================================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"}


@app.route("/")
def index():
    return render_template(
        "index.html",
        device=str(DEVICE),
        ai_model_name=AI_MODEL_NAME,
        ai_ensemble_size=len(ai_ensemble_models),
        ai_folds=ai_ensemble_fold_ids
    )


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "face_morphing_model_loaded": len(face_bundles) > 0,
        "face_morphing_ensemble_size": len(face_bundles),
        "face_morphing_variant": face_variant,
        "face_crop_forced": FACE_FORCE_CROP,
        "audio_model_loaded": audio_model is not None,
        "ai_generated_model_loaded": len(ai_ensemble_models) > 0,
        "ai_generated_model": AI_MODEL_NAME,
        "ai_generated_ensemble_size": len(ai_ensemble_models),
        "ai_generated_folds": ai_ensemble_fold_ids,
        "face_detector_loaded": get_face_detector() is not None,
        "video_ai_generated_weight": VIDEO_AI_GENERATED_WEIGHT,
        "video_face_morph_weight": VIDEO_FACE_MORPH_WEIGHT
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    filepath = None

    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file provided."}), 400

        file = request.files["file"]

        if not file or not file.filename:
            return jsonify({"success": False, "error": "No file selected."}), 400

        analysis_type = request.form.get("type", "face_morph").lower()
        threshold = float(request.form.get("threshold", 0.5))

        use_tta = request.form.get("tta", "false").lower() == "true"
        use_face_crop = request.form.get("face_crop", "false").lower() == "true"
        use_preprocess = request.form.get("preprocess", "false").lower() == "true"

        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        filepath = UPLOAD_DIR / filename
        file.save(str(filepath))

        ext = filepath.suffix.lower()

        if analysis_type in {"ai_generated", "ai", "ai_image"}:
            if ext not in IMAGE_EXTS:
                return jsonify({
                    "success": False,
                    "error": "AI-generated detection requires an image."
                }), 400

            image = Image.open(filepath).convert("RGB")
            result = predict_ai_generated(image)

        elif analysis_type in {"face_morph", "face_morphing", "image"}:
            if ext not in IMAGE_EXTS:
                return jsonify({
                    "success": False,
                    "error": "Face morphing detection requires an image."
                }), 400

            image = Image.open(filepath).convert("RGB")

            result = predict_face_morphing(
                image,
                threshold=threshold,
                use_tta=use_tta,
                use_face_crop=use_face_crop,
                use_preprocess=use_preprocess
            )

        elif analysis_type in {"audio", "audio_deepfake"}:
            if ext not in AUDIO_EXTS:
                return jsonify({
                    "success": False,
                    "error": "Audio deepfake detection requires an audio file."
                }), 400

            result = predict_audio(str(filepath), threshold=threshold)

        elif analysis_type == "video":
            if ext not in VIDEO_EXTS:
                return jsonify({
                    "success": False,
                    "error": "Video detection requires a video file."
                }), 400

            result = predict_video(
                str(filepath),
                threshold=threshold,
                use_tta=use_tta,
                use_face_crop=use_face_crop,
                use_preprocess=use_preprocess
            )

        else:
            return jsonify({
                "success": False,
                "error": f"Unsupported analysis type: {analysis_type}"
            }), 400

        return jsonify({
            "success": True,
            "filename": file.filename,
            **result,
            "device": str(DEVICE),
            "threshold": threshold
        })

    except Exception as exc:
        logger.exception("Prediction failed.")
        return jsonify({"success": False, "error": str(exc)}), 500

    finally:
        if filepath is not None:
            try:
                filepath.unlink(missing_ok=True)
            except Exception:
                pass


@app.errorhandler(413)
def too_large(error):
    return jsonify({
        "success": False,
        "error": "File too large. Maximum size is 100 MB."
    }), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found."}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"success": False, "error": "Internal server error."}), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AUTHENTICAI - COMBINED SYNTHETIC MEDIA DETECTOR")
    print("=" * 70)
    print(f"Device                  : {DEVICE}")
    print(f"Face morphing bundles   : {len(face_bundles)}  (variant: {face_variant})")
    print(f"Audio deepfake model    : {audio_model is not None}")
    print(f"AI-generated model      : {len(ai_ensemble_models) > 0}")
    print(f"AI ensemble folds       : {ai_ensemble_fold_ids}")
    print(f"Video fusion weights    : AI={VIDEO_AI_GENERATED_WEIGHT} "
          f"Face={VIDEO_FACE_MORPH_WEIGHT}")
    print(f"Face detector           : {get_face_detector() is not None}")
    print("=" * 70)

    # use_reloader=False stops debug mode from loading every model twice.
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )