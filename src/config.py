"""
Configuration file for SafetyGuard AI - PPE Detection System
Multi-class edition (no GPU required)
---------------------------------------------------------------------------
Detection now runs TWO lightweight pretrained YOLOv8n models together:

  1. PERSON_MODEL  - stock COCO yolov8n.pt, auto-downloaded by ultralytics.
                      Used only to find where each worker is in the frame.

  2. PPE_MODEL      - a YOLOv8n model already fine-tuned on PPE gear
                      (helmet, vest, gloves, goggles, mask, safety shoe),
                      pulled from the Hugging Face Hub the first time you
                      load the model. No training, no GPU needed.

detection.py then matches each detected PPE item to the nearest detected
person (by box overlap) so violations are reported per-worker: "this
person is missing a helmet", not just "no helmet visible anywhere".
---------------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# Person detector (finds workers in the frame)
# ---------------------------------------------------------------------------
PERSON_MODEL_CONFIG = {
    'weights': 'yolov8n.pt',          # stock COCO model, auto-downloaded
    'confidence_threshold': 0.35,
    'iou_threshold': 0.5,              # tighter NMS - fewer duplicate/merged boxes in crowds
    'imgsz': 640,                      # raise to 960/1280 in Settings for small/distant workers
}

# ---------------------------------------------------------------------------
# PPE / gear detector (finds safety equipment in the frame)
# Pretrained, MIT-licensed, fine-tuned YOLOv8n - no training required.
# Source: https://huggingface.co/Tanishjain9/yolov8n-ppe-detection-6classes
# ---------------------------------------------------------------------------
PPE_MODEL_CONFIG = {
    'hf_repo_id': 'Tanishjain9/yolov8n-ppe-detection-6classes',
    'hf_filename': 'best.pt',
    'local_cache_dir': 'models',
    'local_cache_name': 'ppe_yolov8n_6class.pt',
    'confidence_threshold': 0.25,      # lower - PPE items are small in-frame
    'iou_threshold': 0.45,
    'imgsz': 640,                      # raise to 960/1280 in Settings for small/distant gear
}

# Class names as published by the model (id order matters for reference only;
# detection.py reads the actual names from the loaded model at runtime).
PPE_CLASS_NAMES = ['gloves', 'vest', 'goggles', 'helmet', 'mask', 'safety_shoe']

# Which of the above are treated as REQUIRED for compliance by default.
# Matches the brief: hardhats (helmet), vests, gloves. Goggles/mask/shoe are
# still detected and shown, just not counted as violations unless you enable
# them from the Settings tab.
REQUIRED_CLASSES_DEFAULT = ['helmet', 'vest', 'gloves']
OPTIONAL_CLASSES_DEFAULT = ['goggles', 'mask', 'safety_shoe']

# Per-class overlay colors (BGR, for OpenCV) - used for both required and
# optional gear so every class is visually distinct on the video feed.
PPE_CLASS_COLORS_BGR = {
    'helmet':       (46, 196, 140),   # green
    'vest':         (46, 165, 255),   # amber/orange
    'gloves':       (222, 158, 54),   # blue-teal
    'goggles':      (200, 130, 255),  # pink
    'mask':         (180, 220, 90),   # lime
    'safety_shoe':  (140, 140, 255),  # salmon
}

# ---------------------------------------------------------------------------
# Item <-> person matching (accuracy tuning for crowded / multi-worker scenes)
# ---------------------------------------------------------------------------
MATCH_CONFIG = {
    # How much of a gear item's box must fall inside a person's box to count
    # as theirs. Lower = more forgiving of partial occlusion, but more risk
    # of attributing gear to the wrong nearby person in a tight crowd.
    'containment_threshold': 0.25,
}

# ---------------------------------------------------------------------------
# Video Processing Configuration
# ---------------------------------------------------------------------------
VIDEO_CONFIG = {
    'target_fps': 30,
    'display_width': 1280,
    'display_height': 720,
    'buffer_size': 1,
}

# ---------------------------------------------------------------------------
# Alert / Warning System Configuration
# ---------------------------------------------------------------------------
ALERT_CONFIG = {
    'violation_persistence_frames': 3,   # Consecutive frames before an alert fires (reduces flicker)
    'snapshot_on_violation': True,       # Save an annotated snapshot the moment a new violation starts
    'snapshot_dir': 'logs/violations',
    'sound_alert_default': True,
    'max_log_entries': 200,
}

# ---------------------------------------------------------------------------
# Optional: training configuration, only relevant if you later get access to
# a GPU and want to fine-tune your own model (e.g. on the Hard Hat Workers,
# SH17, or a Roboflow Universe PPE dataset). Not required for normal use -
# the app works out of the box with the pretrained PPE_MODEL above.
# ---------------------------------------------------------------------------
TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 16,
    'image_size': 640,
    'learning_rate': 0.001,
    'patience': 10,
    'device': '0',   # GPU device id, or 'cpu' (very slow for training)
}

DATASET_CONFIG = {
    'data_yaml': 'data/data.yaml',
    'train_path': 'data/train',
    'valid_path': 'data/valid',
    'test_path': 'data/test',
}

# ---------------------------------------------------------------------------
# UI / Theme Configuration - "Safety Amber": deep graphite background with
# amber/orange accents and teal/red status colors.
# ---------------------------------------------------------------------------
UI_CONFIG = {
    'title': 'SafetyGuard AI',
    'refresh_rate': 10,
    'show_fps': True,
    'show_confidence': True,
}

THEME = {
    'bg_start': '#14110F',
    'bg_end': '#1F1A14',
    'panel_start': '#211C17',
    'panel_end': '#161310',
    'border': '#3A2E22',
    'accent': '#FF9F1C',
    'accent_soft': '#FFB84D',
    'accent_dark': '#C97400',
    'success': '#2EC4B6',
    'success_soft': '#7FE8DB',
    'danger': '#E63946',
    'danger_soft': '#FF6B7A',
    'info': '#5E7CE2',
    'text_primary': '#FFF6EC',
    'text_muted': '#B8AA97',
}
