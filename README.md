# AuthenticAI

AuthenticAI is a local multimodal synthetic-media analysis application. It provides a Flask web interface for examining images, audio, and video with specialist deep-learning pipelines and presenting a confidence-based verdict.

> **Important:** AuthenticAI is an analytical aid, not a definitive authenticity judge or legal-evidence system. Results should be reviewed by a qualified human and validated against the context in which the media was created.

## Features

- **Face morphing detection** using a face-cropped EfficientNet/Meso-4 fusion pipeline.
- **AI-generated image detection** using an EfficientNet-B1 five-fold ensemble.
- **Audio deepfake detection** using MFCC-based audio processing and `AudioDeepfakeModel`.
- **Video analysis** using frame sampling and weighted fusion of the image pipelines.
- **Automatic face detection** using the bundled Haar cascade.
- **GPU acceleration** when a CUDA-enabled PyTorch installation is available; otherwise inference runs on the CPU.
- **Temporary uploads** that are deleted after each prediction request.
- **Health endpoint** for checking loaded models, device selection, and ensemble configuration.
- Dark, responsive browser interface with file previews and confidence visualization.

## Technology Stack

- Python
- Flask and Flask-CORS
- PyTorch and TorchVision
- `timm`
- OpenCV
- Librosa
- Pillow
- NumPy
- HTML, CSS, and JavaScript

## Project Structure

```text
.
├── app.py                              # Flask application and inference routes
├── requirements.txt                    # Python dependency pins
├── src/
│   ├── __init__.py
│   └── model.py                        # AudioDeepfakeModel definition
├── templates/
│   └── index.html                      # Browser application
├── static/
│   └── style.css                       # Application styling
├── models/
│   ├── face_cv_v2/                     # Face-morphing model bundles
│   ├── best_audio_deepfake_model.pth   # Audio model checkpoint
│   ├── haarcascade_frontalface_default.xml
│   └── ...                             # Additional model artifacts
├── cv_results_efficientnet_b1_5fold/
│   └── fold_1 ... fold_5/              # AI-image model checkpoints
├── folds/                               # Cross-validation metadata
└── uploads/                             # Runtime upload directory
```

The application expects the trained model artifacts to be present locally. Model files are not downloaded automatically at startup.

## Prerequisites

- Python 3.10 or newer
- Git
- At least 4 GB of available RAM for CPU inference; more may be required for video analysis
- Optional: an NVIDIA GPU with a compatible CUDA-enabled PyTorch installation

For audio processing, ensure the system has a compatible audio backend available to Librosa. On Windows, installing `soundfile` may also be necessary when working with formats that are not handled by the default backend.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ChiragRaut45/AuthenticAi.git
cd AuthenticAi
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
py -m venv venv
venv\Scripts\activate.bat
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The application imports additional runtime packages for computer vision, audio processing, numerical operations, and CORS support. If they are not already included in the environment, install them with:

```bash
python -m pip install flask-cors opencv-contrib-python librosa numpy soundfile
```

`opencv-contrib-python` is preferred because the application uses OpenCV's Haar-cascade APIs. Do not install both `opencv-python` and `opencv-contrib-python` in the same environment; remove the former first if necessary:

```bash
python -m pip uninstall opencv-python -y
python -m pip install opencv-contrib-python
```

## Running the Application

Start the Flask server from the repository root:

```bash
python app.py
```

The server listens on all interfaces at port `5000`. Open:

```text
http://localhost:5000
```

The application starts in debug mode for local development and disables Flask's reloader to avoid loading large model files twice.

### Changing the port

The port is currently defined in the `app.run(...)` block at the bottom of `app.py`. Update it before starting the server if port `5000` is already in use.

## Using the Web Interface

1. Open the application in a browser.
2. Select one of the analysis tabs:
   - Face Morphing
   - AI Generated
   - Audio Deepfake
   - Video
3. Upload or drop a supported file.
4. Select **Analyze exhibit**.
5. Review the verdict, confidence, raw score, model information, and threshold.

Supported formats:

| Analysis type | Formats |
|---|---|
| Face morphing | JPG, JPEG, PNG, GIF, BMP, WEBP |
| AI-generated image | JPG, JPEG, PNG, GIF, BMP, WEBP |
| Audio deepfake | WAV, MP3, M4A, FLAC, OGG |
| Video | MP4, AVI, MOV, MKV, FLV, WEBM |

The maximum request size is **100 MB**.

## API Reference

### Health check

```http
GET /api/health
```

Example:

```bash
curl http://localhost:5000/api/health
```

The response reports the selected device and whether the face, audio, AI-image, and face-detector components loaded successfully.

### Prediction

```http
POST /api/predict
Content-Type: multipart/form-data
```

Required form fields:

- `file`: media file to analyze
- `type`: `face_morph`, `ai_generated`, `audio`, or `video`

Optional form fields:

- `threshold`: numeric decision threshold; defaults to `0.5`
- `tta`: `true` or `false` for test-time augmentation where supported
- `face_crop`: `true` or `false`
- `preprocess`: `true` or `false`

Example image request:

```bash
curl -X POST http://localhost:5000/api/predict ^
  -F "file=@sample.jpg" ^
  -F "type=ai_generated"
```

Example audio request:

```bash
curl -X POST http://localhost:5000/api/predict ^
  -F "file=@sample.wav" ^
  -F "type=audio"
```

Successful responses contain `success: true`, the prediction, confidence, raw score, pipeline metadata, device, and threshold. Invalid files return a JSON error with an appropriate HTTP status code.

## Model Configuration

Model paths and inference settings are defined near the top of [`app.py`](app.py), including:

- Face-morphing bundle directory and selected variant
- Audio checkpoint path
- EfficientNet-B1 fold directory and ensemble size
- Image dimensions and normalization values
- Video fusion weights
- Maximum upload size

If a checkpoint is missing or incompatible, the server logs the loading failure and the health endpoint indicates which model components are unavailable. Confirm that the checkpoint filenames and directory layout match the expected paths before changing inference code.

## Troubleshooting

### Models do not load

- Confirm that all required files under `models/` and `cv_results_efficientnet_b1_5fold/` are present.
- Run `GET /api/health` and inspect the `*_model_loaded` fields.
- Check the terminal output produced during application startup.
- Verify that the installed PyTorch version can load the checkpoint format.

### OpenCV face detection is unavailable

Install the contrib distribution and ensure the bundled cascade file exists:

```bash
python -m pip uninstall opencv-python -y
python -m pip install opencv-contrib-python
```

If no cascade can be loaded, the application falls back to using the full frame.

### CUDA is not being used

Check the startup log for the selected device. If it reports `cpu`, install a PyTorch build compatible with the installed NVIDIA driver and CUDA runtime, then verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Audio analysis fails

- Confirm the input format is supported.
- Install `soundfile` and a compatible audio backend.
- Verify that Librosa and its numerical dependencies are installed in the active virtual environment.

## Development Notes

- Uploaded files are written to `uploads/` with a generated UUID-prefixed filename and removed in the request cleanup path.
- The application uses local inference; no media is sent to a third-party inference service.
- The included `app1.py` is a local or backup application file and is excluded from the primary runtime documentation.
- Large model artifacts can make startup and inference resource-intensive, especially on CPU.

## Responsible Use

Synthetic-media detectors can produce false positives and false negatives. Do not use AuthenticAI as the sole basis for employment, financial, legal, safety, or other high-impact decisions. Preserve the original file and chain of custody, record the model configuration used, and obtain independent review for consequential investigations.

## License

No license file is currently included in this repository. Add an explicit license before distributing the project or its model artifacts.
