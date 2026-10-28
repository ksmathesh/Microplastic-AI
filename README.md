# AquaSentinel AI
**Image-based screening for suspected microplastic particles in prepared water samples.**

This semester-project prototype applies YOLO object detection to microscope or camera images. The web workspace supports image upload, detection visualization, class counts, confidence, and saved analysis history. Results are **suspected particles** based on image appearance; images alone cannot chemically identify plastic.

## Current scope
- FastAPI service, SQLite local analysis history, responsive web dashboard.
- YOLOv8 fine-tuning entry point for the available four-class dataset.
- The current YOLO checkpoint is available at `weights/best.pt`; predictions are image-based screening results, not chemical confirmation.
- IoT sensors, physical filtration hardware, and trend prediction are outside this first software build.

The dataset metadata reports 6,003 images and CC BY 4.0 license. Read data/README.md. The dataset is excluded from Git.

## Run locally in PowerShell

Use Python 3.12 for the local FastAPI environment:

    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements-web.txt
    Copy-Item .env.example .env
    python -m uvicorn backend.app:app --reload

Open http://127.0.0.1:8000. API docs: http://127.0.0.1:8000/docs. Without a checkpoint the dashboard/API still runs, but inference reports that a trained model is required.

## Train the model

The local CPU PyTorch package is large. Use `notebooks/train_on_colab.ipynb` for the first baseline. For a separate accuracy experiment, use `notebooks/train_accuracy_experiment_colab.ipynb`; it audits labels, validates the current checkpoint on the validation split, optionally adds 1,406 training images from a CC BY 4.0 wastewater microscopy dataset to the existing Fragment class, trains a YOLOv8s candidate at 960 pixels, compares confidence thresholds on the original validation set, and saves the candidate and metrics to Drive without overwriting the current checkpoint. The external source has different imaging conditions and is added only to training; inspect predictions and retain it only if validation improves. The Colab notebook downloads this external training split (about 0.9 GB) directly when enabled. Colab GPU and internet access are required. Keep the current model until the candidate is compared and evaluated.

For local training, install the full model stack (`python -m pip install -r requirements.txt`) and run:

    python train.py --data "<path-to-dataset>\data.yaml" --epochs 50 --imgsz 640 --batch 8

The training script normalizes the Roboflow split paths without modifying the source dataset. Set `MODEL_PATH=weights/best.pt` in `.env` after copying the trained checkpoint, then restart the API.
## Evaluation
Use the held-out test split. Record checkpoint, training config, dataset version, per-class precision/recall, mAP@0.5 and mAP@0.5:0.95, confusion matrix, and false-positive/negative examples. Do not report training scores as final results or claim chemical confirmation/general performance without evidence.

## Deployment
This is a local starter, not published or hosted. Hosting needs persistent database and image storage, a production model artifact, upload limits, and confirmation the provider supports model compute requirements. Keep credentials and the full dataset out of Git; preserve dataset attribution. The supplied `weights/best.pt` checkpoint is included for local inference; other training outputs remain excluded.

