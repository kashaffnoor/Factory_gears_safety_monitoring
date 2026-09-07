# SafetyGuard AI — PPE / Safety Compliance Monitoring

Real-time detection of whether workers on a factory floor are wearing required
safety equipment (**hardhat, vest, gloves** — plus goggles/mask/shoe as
optional extras), using camera footage, with violations flagged live.
**No GPU or training required** — it runs on CPU out of the box.

## How it works

Two lightweight, pretrained YOLOv8n models run together on every frame:

| Model | Role | Source |
|---|---|---|
| **Person detector** | Finds each worker in the frame | Stock `yolov8n.pt` (COCO), auto-downloaded by `ultralytics` |
| **PPE gear detector** | Finds helmet / vest / gloves / goggles / mask / safety_shoe | [`Tanishjain9/yolov8n-ppe-detection-6classes`](https://huggingface.co/Tanishjain9/yolov8n-ppe-detection-6classes) (MIT license), fine-tuned on a Roboflow-format multi-class PPE dataset — pulled from the Hugging Face Hub the first time you click **Load Model**, then cached locally |

Every detected gear item is matched to the person box it belongs to. In
crowded scenes where workers stand close together, matching uses **how much
of the item overlaps each person** (not raw IoU — a helmet is always tiny
relative to a full body box) and breaks ties by **which person's box-center
is nearest**, so a helmet doesn't get incorrectly attributed to the person
standing next to its actual owner. A worker is flagged as a **violation** if
any of your checked "required" items aren't found on them. Alerts only fire
once a violation persists for a few consecutive frames, so a single missed
detection doesn't cause a false alarm.

This satisfies the original brief's model/software stack:
- **Model:** YOLOv8 (Ultralytics), pretrained checkpoint used as-is — no
  fine-tuning needed since it already covers hardhat/vest/glove/goggle/mask/shoe.
- **Software stack:** `ultralytics` for inference, Hugging Face Hub for
  pulling the PPE checkpoint, OpenCV for the camera/video pipeline, and
  Streamlit for the live dashboard with alerts.

## Features

- Real-time detection of workers **and** PPE gear, matched per-worker
- Accurate crowd handling: gear-to-worker matching uses containment +
  nearest-distance tie-breaking so overlapping/close-together workers each
  get their own correct helmet/vest/glove, not a neighbor's
- Adjustable detection resolution (640/960/1280) for better recall on small
  or distant workers in group shots
- Collision-free on-screen labels — violation tags no longer overlap when
  workers stand close together
- Webcam or uploaded video file input
- Checklist of which gear is "required" (default: helmet, vest, gloves) —
  adjustable live from Settings
- Debounced (flicker-free) visual **and audible** alerts
- Automatic timestamped snapshot logging of confirmed violations
- Live compliance statistics and a compliance-rate trend chart
- Clean two-column dashboard: source/controls on the left, live video on the
  right, with Analytics / Violation Log / Settings as separate tabs

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run src/app.py
```

Or on Windows, just run `run_app.bat`.

In the sidebar: click **Load Model** (this downloads the ~6MB PPE checkpoint
from Hugging Face the first time — needs internet once, then it's cached in
`models/` and works offline), pick **Webcam** or upload a video, then click
**Start**.

## Usage

1. Open the web interface (default: http://localhost:8501)
2. Click **Load Model** in the sidebar and wait for both models to load
3. Optionally open **Settings** to choose which gear is required and tune
   sensitivity/debounce
4. Choose **Webcam** or upload a **video file**
5. Click **Start**
6. **Live Feed** shows boxes + FPS + alerts; **Analytics** shows the
   compliance trend; **Violation Log** shows captured incidents

## Configuration

Defaults live in `src/config.py`:

- `PERSON_MODEL_CONFIG` / `PPE_MODEL_CONFIG` — model sources + confidence/IOU thresholds
- `REQUIRED_CLASSES_DEFAULT` — `['helmet', 'vest', 'gloves']` (matches the brief)
- `OPTIONAL_CLASSES_DEFAULT` — `['goggles', 'mask', 'safety_shoe']` (detected, shown, not required by default)
- `ALERT_CONFIG` — debounce length, snapshot/sound behavior

## Project Structure

```
Safety Compliance Monitoring/
├── models/                # Cached PPE model weights (downloaded on first run)
├── logs/
│   └── violations/        # Auto-captured violation snapshots
├── src/
│   ├── app.py              # Streamlit GUI
│   ├── detection.py         # Dual-model detection + per-worker matching
│   ├── train.py               # Optional: fine-tune your own model (needs GPU)
│   ├── config.py                # Configuration + theme
│   └── utils.py                   # Utility functions + embedded alert tone
├── data/                   # Only needed if you train your own model
│   └── data.yaml
├── requirements.txt
└── README.md
```

## If you later get access to a GPU

The bundled `train.py` and `data/data.yaml` can fine-tune your own model on
the **Hard Hat Workers dataset (Kaggle)** already referenced in this project,
or you can swap in the **SH17 dataset (Hugging Face)** or a **Roboflow
Universe "PPE Detection"** export for broader multi-class coverage (vests,
gloves, goggles). Update `data/data.yaml` with your classes, then:

```bash
python src/train.py
```

This is entirely optional — the app works fully without it.

## Troubleshooting

**"Could not load the PPE gear model" warning:** the Hugging Face download
failed (no internet, or huggingface.co is blocked on your network). The app
falls back to person-only detection until you retry **Load Model** with a
working connection.

**Webcam not working:** check camera permissions, close other apps using the
camera, try a different **Webcam device index** in Settings.

**Low FPS:** running two models per frame on CPU is heavier than one — watch
the FPS chip on Live Feed; lower the display resolution in `config.py` or
raise the confidence thresholds to reduce false-positive box drawing overhead.

**Too many/few alerts:** raise or lower the alert debounce in Settings.

## Requirements

- Python 3.8+
- Windows, Linux, or macOS
- Webcam (for live detection)
- Internet connection on first run only (to download both models)
- No GPU required

## License

Built on:
- Ultralytics YOLOv8 (AGPL-3.0)
- `Tanishjain9/yolov8n-ppe-detection-6classes` (MIT)
- OpenCV (Apache 2.0)
- Streamlit (Apache 2.0)
- Hugging Face Hub (Apache 2.0)
