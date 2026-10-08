import os
import sys
import io
import uuid
import logging
import traceback
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

from PIL import Image, ImageEnhance, ImageOps, ImageFile, UnidentifiedImageError
from torchvision import transforms, models

# ============================================================
# AUTHENTICAI - COMBINED DETECTOR
#
# Pipelines:
#   1. Face Morphing  -> EfficientNet-B0 + MesoNet fusion
#   2. AI Generated   -> EfficientNet-B1 5-fold ensemble
#   3. Audio Deepfake  -> AudioDeepfakeModel
#   4. Video          -> per-frame fusion of (1) + (2), weighted
#                         toward the AI-generated ensemble, then
#                         soft-voted across frames
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
# MODEL PATHS
# ============================================================

FACE_EFF_PATH = MODELS_DIR / "best_model.pth"
FACE_MESO_PATH = MODELS_DIR / "mesonet_best.pth"
FACE_FUSION_PATH = MODELS_DIR / "fusion_model.pth"

AUDIO_PATH = MODELS_DIR / "best_audio_deepfake_model.pth"

# The file best_model-v3.pt is kept in models/, but it is NOT used
# by the AI-generated pipeline below. The AI-generated pipeline
# supplied by the user is the 5-fold EfficientNet-B1 ensemble.
AI_MODEL_NAME = "efficientnet_b1"
AI_IMAGE_SIZE = 240
AI_NUM_FOLDS = 5
AI_DEFAULT_CLASSES = ["Real", "Fake"]

# ---- video fusion weights -----------------------------------
# Per frame, the video pipeline blends the AI-generated ensemble's
# fake-probability with the face-morphing fusion model's fake-
# probability. The AI-generated ensemble is given the majority
# share of the vote by default. Tune these two (they should sum
# to 1.0) to shift how much each specialist influences the verdict.
VIDEO_AI_GENERATED_WEIGHT = 0.65
VIDEO_FACE_MORPH_WEIGHT = 0.35

# ============================================================
# COMMON IMAGE TRANSFORM
# ============================================================

FACE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

FACE_TTA_TRANSFORMS = [
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(saturation=0.2, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
]

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
# Uses OpenCV's classic Haar-cascade face detector. As of OpenCV
# 5.0, cv2.CascadeClassifier was moved out of the core module into
# opencv_contrib, so this requires `opencv-contrib-python` rather
# than plain `opencv-python`:
#
#     pip uninstall opencv-python -y
#     pip install opencv-contrib-python
#
# The cascade XML is bundled locally (models/haarcascade_frontal
# face_default.xml) so this doesn't depend on cv2.data pointing to
# the right place in whatever environment this runs in; it falls
# back to cv2's own bundled copy if that file isn't present.
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
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    if not os.path.exists(cascade_path):
        logger.warning("Haar Cascade file not found. Full frames will be used.")
        return None

    try:
        classifier = cv2.CascadeClassifier(cascade_path)
        if classifier.empty():
            logger.warning("Haar Cascade file found but failed to load: %s", cascade_path)
            return None
        _face_detector = classifier
        logger.info("Haar Cascade face detector loaded from: %s", cascade_path)
    except AttributeError:
        logger.warning(
            "cv2.CascadeClassifier is unavailable — you likely have plain "
            "opencv-python installed instead of opencv-contrib-python. Run: "
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
# ============================================================

class MesoNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 8, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)

        self.conv3 = nn.Conv2d(8, 16, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 16, 5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(16 * 14 * 14, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))

        return self.fc2(x)


class FusionModel(nn.Module):
    """
    Face-morphing vision fusion from the supplied application:
    EfficientNet-B0 + MesoNet.
    """

    def __init__(self, eff, meso):
        super().__init__()

        self.eff = eff
        self.meso = meso

        self.eff.classifier = nn.Identity()
        self.meso.fc2 = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(1280 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        f1 = self.eff(x)
        f2 = self.meso(x)

        return self.fc(torch.cat((f1, f2), dim=1))


# ============================================================
# LOAD FACE MORPHING + AUDIO MODELS
# ============================================================

fusion_model = None
audio_model = None


def load_face_and_audio_models():
    global fusion_model, audio_model

    # ----- Face morphing vision model -----
    required = [FACE_EFF_PATH, FACE_MESO_PATH, FACE_FUSION_PATH]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing face-morphing model: {path}")

    eff_model = models.efficientnet_b0(weights=None)
    eff_model.classifier[1] = nn.Linear(
        eff_model.classifier[1].in_features,
        2
    )

    eff_state = torch.load(
        FACE_EFF_PATH,
        map_location=DEVICE,
        weights_only=False
    )
    eff_model.load_state_dict(eff_state)
    eff_model.to(DEVICE).eval()

    meso_model = MesoNet()

    meso_state = torch.load(
        FACE_MESO_PATH,
        map_location=DEVICE,
        weights_only=False
    )
    meso_model.load_state_dict(meso_state)
    meso_model.to(DEVICE).eval()

    fusion_model = FusionModel(eff_model, meso_model).to(DEVICE)

    fusion_state = torch.load(
        FACE_FUSION_PATH,
        map_location=DEVICE,
        weights_only=False
    )
    fusion_model.load_state_dict(fusion_state)
    fusion_model.eval()

    logger.info("Face morphing fusion model loaded.")

    # ----- Audio model -----
    if AUDIO_PATH.exists():
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
    else:
        logger.warning("Audio model file not found: %s", AUDIO_PATH)


# ============================================================
# AI-GENERATED IMAGE MODEL
# Exact 5-fold EfficientNet-B1 approach supplied by the user.
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
        logger.warning(
            "Missing keys in %s: %s",
            model_path.name,
            missing
        )

    if unexpected:
        logger.warning(
            "Unexpected keys in %s: %s",
            model_path.name,
            unexpected
        )

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
            logger.warning(
                "AI fold %d missing: %s",
                fold,
                model_path
            )
            continue

        try:
            model, local_classes = load_ai_checkpoint(model_path)

            ai_ensemble_models.append(model)
            ai_ensemble_fold_ids.append(fold)

            if resolved_classes is None:
                resolved_classes = local_classes

            logger.info("AI-generated fold %d loaded.", fold)

        except Exception as exc:
            logger.exception(
                "AI-generated fold %d failed: %s",
                fold,
                exc
            )

    if not ai_ensemble_models:
        raise RuntimeError(
            "No AI-generated EfficientNet-B1 fold models could be loaded."
        )

    ai_class_names = resolved_classes or AI_DEFAULT_CLASSES.copy()

    lower_names = [name.lower() for name in ai_class_names]

    ai_fake_index = (
        lower_names.index("fake")
        if "fake" in lower_names
        else 1
    )

    ai_real_index = (
        lower_names.index("real")
        if "real" in lower_names
        else 0
    )

    logger.info(
        "AI-generated ensemble loaded: folds=%s classes=%s",
        ai_ensemble_fold_ids,
        ai_class_names
    )


# Load all models when Flask starts.
load_face_and_audio_models()
load_ai_models()


# ============================================================
# FACE MORPHING PREDICTION
# ============================================================

def predict_face_morphing(
    image,
    threshold=0.5,
    use_tta=False,
    use_face_crop=False,
    use_preprocess=False
):
    if fusion_model is None:
        raise RuntimeError("Face morphing model is not loaded.")

    image = image.convert("RGB")

    if use_preprocess:
        image = normalize_image_quality(image)

    if use_face_crop:
        image = detect_and_crop_face(image)

    aug_list = FACE_TTA_TRANSFORMS if use_tta else [FACE_TRANSFORM]

    scores = []

    with torch.inference_mode():
        for aug in aug_list:
            tensor = aug(image).unsqueeze(0).to(DEVICE)

            output = fusion_model(tensor)
            probabilities = torch.softmax(output, dim=1)

            # Preserves the supplied face-morphing app's score convention:
            # class index 1 is used as raw_score.
            scores.append(float(probabilities[0][1].cpu()))

    raw_score = float(np.mean(scores))

    # Preserve the original supplied application's prediction convention.
    prediction = "Real" if raw_score >= threshold else "Fake"
    confidence = max(raw_score, 1 - raw_score) * 100

    return {
        "prediction": prediction,
        "label": "FACE MORPHING" if prediction == "Fake" else "REAL",
        "confidence": round(confidence, 2),
        "raw_score": round(raw_score, 4),
        "pipeline": "Face Morphing",
        "tta_enabled": use_tta,
        "face_crop_enabled": use_face_crop,
        "preprocess_enabled": use_preprocess
    }


# ============================================================
# AUDIO DEEPFAKE
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

    features = np.concatenate(
        [mfcc, mfcc_delta, mfcc_delta2],
        axis=0
    )

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

    features_resized = cv2.resize(
        features,
        (224, 224)
    )

    f_min = features_resized.min()
    f_max = features_resized.max()

    if f_max > f_min:
        features_resized = (
            features_resized - f_min
        ) / (f_max - f_min)

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
#   - the face-morphing fusion model (face-swap / morph artifacts)
#   - the AI-generated 5-fold EfficientNet-B1 ensemble (fully
#     synthetic / generated frames)
#
# The two per-frame fake-probabilities are blended with the AI-
# generated ensemble holding the majority weight (see
# VIDEO_AI_GENERATED_WEIGHT / VIDEO_FACE_MORPH_WEIGHT above), then
# the video-level verdict is a soft vote (average fused probability,
# thresholded once) rather than a hard per-frame majority vote.
# ============================================================

def extract_frames(video_path, max_frames=20):
    frames = []

    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)

    count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if count % step == 0:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
            frames.append(Image.fromarray(frame))

        count += 1

    cap.release()

    return frames


def _face_morph_fake_probability(
    frame,
    use_tta,
    use_face_crop,
    use_preprocess
):
    """
    Runs the face-morphing fusion model on a single frame and returns
    its fake-probability (0-1), independent of any threshold, so it
    can be safely blended with the AI-generated ensemble's score.
    """
    result = predict_face_morphing(
        frame,
        threshold=0.5,
        use_tta=use_tta,
        use_face_crop=use_face_crop,
        use_preprocess=use_preprocess
    )
    # predict_face_morphing's raw_score follows the "higher = more
    # real" convention (see comment in that function), so flip it.
    return 1.0 - result["raw_score"]


def _ai_generated_fake_probability(frame):
    """
    Runs the AI-generated 5-fold ensemble on a single frame and
    returns its fake-probability (0-1).
    """
    result = predict_ai_generated(frame)
    # predict_ai_generated's raw_score is already P(fake).
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

    # Soft vote: average the fused per-frame probability once, then
    # threshold that single number, instead of averaging confidences
    # of frames that may have disagreed with each other.
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
# AI-GENERATED IMAGE PREDICTION
# Exact logic from the supplied EfficientNet-B1 5-fold app.
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

        probabilities = (
            summed_probabilities / len(ai_ensemble_models)
        )

        predicted_index = int(
            torch.argmax(probabilities).item()
        )

    predicted_class = ai_class_names[predicted_index]

    fake_probability = float(
        probabilities[ai_fake_index].item() * 100
    )

    real_probability = float(
        probabilities[ai_real_index].item() * 100
    )

    confidence = float(
        probabilities[predicted_index].item() * 100
    )

    label = (
        "AI GENERATED"
        if predicted_index == ai_fake_index
        else "REAL"
    )

    return {
        "prediction": (
            "Fake"
            if predicted_index == ai_fake_index
            else "Real"
        ),
        "label": label,
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
        "face_morphing_model_loaded": fusion_model is not None,
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
            return jsonify({
                "success": False,
                "error": "No file provided."
            }), 400

        file = request.files["file"]

        if not file or not file.filename:
            return jsonify({
                "success": False,
                "error": "No file selected."
            }), 400

        analysis_type = request.form.get(
            "type",
            "face_morph"
        ).lower()

        threshold = float(
            request.form.get("threshold", 0.5)
        )

        use_tta = (
            request.form.get("tta", "false").lower()
            == "true"
        )

        use_face_crop = (
            request.form.get("face_crop", "false").lower()
            == "true"
        )

        use_preprocess = (
            request.form.get("preprocess", "false").lower()
            == "true"
        )

        filename = (
            f"{uuid.uuid4().hex}_"
            f"{secure_filename(file.filename)}"
        )

        filepath = UPLOAD_DIR / filename
        file.save(str(filepath))

        ext = filepath.suffix.lower()

        # ----------------------------------------------------
        # AI-GENERATED IMAGE
        # ----------------------------------------------------
        if analysis_type in {
            "ai_generated",
            "ai",
            "ai_image"
        }:
            if ext not in {
                ".jpg", ".jpeg", ".png",
                ".gif", ".bmp", ".webp"
            }:
                return jsonify({
                    "success": False,
                    "error": "AI-generated detection requires an image."
                }), 400

            image = Image.open(filepath).convert("RGB")
            result = predict_ai_generated(image)

        # ----------------------------------------------------
        # FACE MORPHING IMAGE
        # ----------------------------------------------------
        elif analysis_type in {
            "face_morph",
            "face_morphing",
            "image"
        }:
            if ext not in {
                ".jpg", ".jpeg", ".png",
                ".gif", ".bmp", ".webp"
            }:
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

        # ----------------------------------------------------
        # AUDIO DEEPFAKE
        # ----------------------------------------------------
        elif analysis_type in {
            "audio",
            "audio_deepfake"
        }:
            if ext not in {
                ".wav", ".mp3", ".m4a",
                ".flac", ".ogg"
            }:
                return jsonify({
                    "success": False,
                    "error": "Audio deepfake detection requires an audio file."
                }), 400

            result = predict_audio(
                str(filepath),
                threshold=threshold
            )

        # ----------------------------------------------------
        # VIDEO
        # ----------------------------------------------------
        elif analysis_type == "video":
            if ext not in {
                ".mp4", ".avi", ".mov",
                ".mkv", ".flv", ".webm"
            }:
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

        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500

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
    return jsonify({
        "success": False,
        "error": "Endpoint not found."
    }), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error."
    }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AUTHENTICAI - COMBINED SYNTHETIC MEDIA DETECTOR")
    print("=" * 70)
    print(f"Device                  : {DEVICE}")
    print(f"Face morphing model     : {fusion_model is not None}")
    print(f"Audio deepfake model    : {audio_model is not None}")
    print(f"AI-generated model      : {len(ai_ensemble_models) > 0}")
    print(f"AI ensemble folds       : {ai_ensemble_fold_ids}")
    print(f"Video fusion weights    : AI={VIDEO_AI_GENERATED_WEIGHT} "
          f"Face={VIDEO_FACE_MORPH_WEIGHT}")
    print(f"Face detector           : {get_face_detector() is not None}")
    print("=" * 70)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )