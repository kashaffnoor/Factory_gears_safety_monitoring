"""
Streamlit GUI for SafetyGuard AI - PPE Detection System
Multi-class edition (no GPU / training required)
---------------------------------------------------------------------------
Real-time webcam / video PPE detection with live, per-worker violation
warnings. Detection runs two pretrained YOLOv8n models together (a stock
person detector + a PPE-item detector already fine-tuned on helmet, vest,
gloves, goggles, mask, safety_shoe) - see detection.py and config.py. No
training or GPU is required; the PPE model is pulled from the Hugging Face
Hub the first time you click "Load Model" and cached locally after that.

Interface highlights:
  * "Safety Amber" color theme (graphite + amber/teal).
  * Tabbed layout: Live Feed / Analytics / Violation Log / Settings.
  * Live FPS + inference-time + people/items-count chips over the feed.
  * Debounced alerts drive a pulsing on-screen banner AND an audible beep,
    firing once per incident instead of every frame.
  * Every confirmed violation is auto-captured as a timestamped snapshot,
    browsable in the Violation Log tab.
  * A live compliance-rate trend chart in the Analytics tab.
  * Required-gear checklist (helmet/vest/gloves/...) and alert debounce are
    adjustable from Settings, applied the next time detection starts.
---------------------------------------------------------------------------
"""

import streamlit as st
import cv2
import numpy as np
import time
from pathlib import Path
from collections import deque
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
    st.stop()

try:
    from detection import PPEDetector
    from config import (
        PERSON_MODEL_CONFIG, PPE_MODEL_CONFIG, PPE_CLASS_NAMES,
        REQUIRED_CLASSES_DEFAULT, OPTIONAL_CLASSES_DEFAULT, PPE_CLASS_COLORS_BGR,
        VIDEO_CONFIG, UI_CONFIG, ALERT_CONFIG, THEME, MATCH_CONFIG,
    )
    from utils import save_violation_snapshot, ALERT_SOUND_B64
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
    st.stop()

T = THEME  # shorthand

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Safety Compliance Monitoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme CSS - "Safety Amber": deep graphite background, amber/orange accent,
# teal for compliant, red for violations. Deliberately different palette
# from the previous blue industrial theme.
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
    .stApp {{
        background: radial-gradient(circle at 15% 0%, {T['bg_end']} 0%, {T['bg_start']} 55%);
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {T['panel_start']} 0%, {T['bg_start']} 100%);
        border-right: 1px solid {T['border']};
    }}

    .sg-header {{
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, {T['accent_soft']} 0%, {T['accent']} 55%, {T['accent_dark']} 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 0.1rem;
    }}

    .sg-subtitle {{
        font-size: 0.95rem;
        color: {T['text_muted']};
        margin-bottom: 1.1rem;
    }}

    .sg-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: {T['panel_start']};
        border: 1px solid {T['border']};
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        color: {T['text_primary']};
        margin-right: 8px;
        margin-bottom: 6px;
    }}

    .sg-dot {{
        width: 8px; height: 8px; border-radius: 50%;
        display: inline-block;
    }}

    .sg-card {{
        background: linear-gradient(160deg, {T['panel_start']} 0%, {T['panel_end']} 100%);
        border: 1px solid {T['border']};
        border-radius: 14px;
        padding: 1rem 1.1rem;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .sg-card:hover {{
        transform: translateY(-2px);
        border-color: {T['accent_dark']};
    }}

    .sg-card-label {{
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {T['text_muted']};
        margin-bottom: 0.3rem;
    }}

    .sg-card-value {{
        font-size: 1.9rem;
        font-weight: 800;
        color: {T['text_primary']};
        line-height: 1.1;
    }}

    .sg-progress-track {{
        background: {T['bg_start']};
        border-radius: 999px;
        height: 6px;
        overflow: hidden;
        margin-top: 0.5rem;
    }}
    .sg-progress-fill {{ height: 100%; border-radius: 999px; }}

    @keyframes sg-pulse {{
        0%   {{ box-shadow: 0 0 0 0 rgba(230, 57, 70, 0.55); }}
        70%  {{ box-shadow: 0 0 0 14px rgba(230, 57, 70, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(230, 57, 70, 0); }}
    }}

    .sg-alert {{
        background: linear-gradient(135deg, #3A0E13 0%, #260709 100%);
        border: 2px solid {T['danger']};
        border-radius: 14px;
        padding: 1rem 1.2rem;
        animation: sg-pulse 1.4s infinite;
        margin: 0.6rem 0;
    }}
    .sg-alert-title {{
        font-size: 1.15rem;
        font-weight: 800;
        color: #FFE1E4;
    }}
    .sg-alert-details {{
        font-size: 0.85rem;
        color: {T['danger_soft']};
        margin-top: 2px;
    }}

    .sg-ok {{
        background: linear-gradient(135deg, #0E2E2A 0%, #081E1B 100%);
        border: 2px solid {T['success']};
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
    }}
    .sg-ok-title {{ font-size: 1.05rem; font-weight: 700; color: {T['text_primary']}; }}
    .sg-ok-details {{ font-size: 0.82rem; color: {T['success_soft']}; }}

    .sg-info {{
        background: linear-gradient(135deg, #191F3A 0%, #10142A 100%);
        border: 2px solid {T['info']};
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
    }}
    .sg-info-title {{ font-size: 1.05rem; font-weight: 700; color: {T['text_primary']}; }}
    .sg-info-details {{ font-size: 0.82rem; color: {T['text_muted']}; }}

    .sg-video-frame {{
        border: 2px solid {T['border']};
        border-radius: 14px;
        overflow: hidden;
        padding: 6px;
        background: {T['panel_end']};
    }}

    .sg-chip-row {{ margin-bottom: 0.6rem; }}

    [data-testid="stSidebar"] .stButton > button {{
        border-radius: 10px;
        font-weight: 600;
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(90deg, {T['accent']} 0%, {T['accent_dark']} 100%);
        border: none;
        color: #1A1200;
    }}

    .sg-section {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {T['text_primary']};
        margin: 0.4rem 0 0.6rem 0;
        border-left: 4px solid {T['accent']};
        padding-left: 10px;
    }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def initialize_session_state():
    defaults = {
        'detector': None,
        'processing': False,
        'stats': {'total_frames': 0, 'total_detections': 0, 'violations': 0, 'compliant_detections': 0},
        'video_source': None,
        'model_loaded_time': None,
        'frame_count': 0,
        'confidence': PPE_MODEL_CONFIG['confidence_threshold'],
        'person_confidence': PERSON_MODEL_CONFIG['confidence_threshold'],
        'debounce_frames': ALERT_CONFIG['violation_persistence_frames'],
        'sound_alert': ALERT_CONFIG.get('sound_alert_default', True),
        'camera_index': 0,
        'required_classes': list(REQUIRED_CLASSES_DEFAULT),
        'compliance_history': deque(maxlen=120),   # (elapsed_seconds, compliance_pct)
        'violation_log': deque(maxlen=ALERT_CONFIG.get('max_log_entries', 200)),
        'fps_window': deque(maxlen=30),
        'session_start': None,
        'last_alert_audio_key': 0,
        'ppe_model_status': None,   # None | 'ok' | 'unavailable'
        'detection_resolution': PERSON_MODEL_CONFIG.get('imgsz', 640),
        'match_threshold': MATCH_CONFIG.get('containment_threshold', 0.25),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _load_person_model():
    """Stock COCO YOLOv8n - ultralytics auto-downloads it on first use."""
    return YOLO(PERSON_MODEL_CONFIG['weights'])


def _load_ppe_model():
    """
    Downloads (once) and loads the pretrained multi-class PPE model from the
    Hugging Face Hub: helmet / vest / gloves / goggles / mask / safety_shoe.
    Cached locally afterwards - subsequent loads are instant and offline.
    Returns None (with a UI warning) if the download fails, e.g. no internet.
    """
    cache_dir = Path(PPE_MODEL_CONFIG['local_cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / PPE_MODEL_CONFIG['local_cache_name']

    if not cache_path.exists():
        downloaded_path = hf_hub_download(
            repo_id=PPE_MODEL_CONFIG['hf_repo_id'],
            filename=PPE_MODEL_CONFIG['hf_filename'],
        )
        import shutil
        shutil.copy(downloaded_path, cache_path)

    return YOLO(str(cache_path))


def load_model():
    draw_colors = {
        'ok': (140, 196, 46),
        'violation': (58, 57, 230),
        'unknown': (200, 200, 200),
    }
    try:
        with st.spinner("Loading person detector (yolov8n.pt)..."):
            person_model = _load_person_model()
    except Exception as e:
        st.error(f"Failed to load person detector: {e}")
        return False

    ppe_model = None
    try:
        with st.spinner("Downloading / loading PPE model from Hugging Face (first run only)..."):
            ppe_model = _load_ppe_model()
        st.session_state.ppe_model_status = 'ok'
    except Exception as e:
        st.session_state.ppe_model_status = 'unavailable'
        st.warning(
            f"Could not load the PPE gear model ({e}). "
            "Falling back to person-only detection - no compliance checking. "
            "Check your internet connection and click 'Load Model' again."
        )

    cfg = {
        'PERSON_MODEL_CONFIG': dict(PERSON_MODEL_CONFIG, confidence_threshold=st.session_state.person_confidence,
                                     imgsz=st.session_state.detection_resolution),
        'PPE_MODEL_CONFIG': dict(PPE_MODEL_CONFIG, confidence_threshold=st.session_state.confidence,
                                  imgsz=st.session_state.detection_resolution),
        'REQUIRED_CLASSES': st.session_state.required_classes,
        'ALERT_CONFIG': dict(ALERT_CONFIG, violation_persistence_frames=st.session_state.debounce_frames),
        'PPE_CLASS_COLORS_BGR': PPE_CLASS_COLORS_BGR,
        'MATCH_CONFIG': dict(MATCH_CONFIG, containment_threshold=st.session_state.match_threshold),
    }
    st.session_state.detector = PPEDetector(person_model, ppe_model, cfg, draw_colors=draw_colors)
    st.session_state.model_loaded_time = time.time()

    if ppe_model is not None:
        st.success("Models loaded: person detector + PPE gear detector (helmet, vest, gloves, goggles, mask, shoe).")
    return True


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_status_chips(fps, frame_stats):
    dot = f"<span class='sg-dot' style='background:{T['success']}'></span>" if st.session_state.processing \
        else f"<span class='sg-dot' style='background:{T['text_muted']}'></span>"
    chips = (
        f"<div class='sg-chip-row'>"
        f"<span class='sg-pill'>{dot} {'LIVE' if st.session_state.processing else 'IDLE'}</span>"
        f"<span class='sg-pill'>⚡ {fps:.1f} FPS</span>"
        f"<span class='sg-pill'>🧮 {frame_stats.get('infer_ms', 0):.0f} ms/frame</span>"
        f"<span class='sg-pill'>👷 {frame_stats.get('people_detected', 0)} people</span>"
        f"<span class='sg-pill'>🦺 {frame_stats.get('items_detected', 0)} gear items</span>"
        f"</div>"
    )
    st.markdown(chips, unsafe_allow_html=True)


def render_alert_banner(frame_stats, placeholder, audio_placeholder):
    with placeholder.container():
        if frame_stats.get('alert_active'):
            gear = ', '.join(sorted(set(frame_stats.get('missing_gear', ['helmet'])))) or 'helmet'
            st.markdown(f"""
            <div class="sg-alert">
                <div class="sg-alert-title">⚠️ SAFETY VIOLATION - {frame_stats.get('violations', 0)} worker(s) unprotected</div>
                <div class="sg-alert-details">Missing: {gear} &nbsp;•&nbsp; alert persists until gear is detected</div>
            </div>
            """, unsafe_allow_html=True)
        elif not st.session_state.detector or not frame_stats.get('ppe_model_available'):
            st.markdown("""
            <div class="sg-info">
                <div class="sg-info-title">🎯 Detection Active</div>
                <div class="sg-info-details">General object detection mode (load a custom PPE model for compliance alerts)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sg-ok">
                <div class="sg-ok-title">✅ All Clear</div>
                <div class="sg-ok-details">Every worker in frame has the required safety gear</div>
            </div>
            """, unsafe_allow_html=True)

    # Fire an audible beep exactly once per new violation event
    if frame_stats.get('new_violation_event') and st.session_state.sound_alert:
        st.session_state.last_alert_audio_key += 1
        key = st.session_state.last_alert_audio_key
        audio_placeholder.markdown(
            f"""<audio autoplay="true" key="{key}">
                    <source src="data:audio/wav;base64,{ALERT_SOUND_B64}" type="audio/wav">
                </audio>""",
            unsafe_allow_html=True,
        )
    elif not frame_stats.get('alert_active'):
        audio_placeholder.empty()


def log_violation_snapshot(annotated_frame, frame_stats):
    if not ALERT_CONFIG.get('snapshot_on_violation', True):
        return
    path, ts = save_violation_snapshot(annotated_frame, ALERT_CONFIG.get('snapshot_dir', 'logs/violations'))
    if path:
        st.session_state.violation_log.appendleft({
            'path': path,
            'timestamp': ts,
            'missing_gear': list(dict.fromkeys(frame_stats.get('missing_gear', ['helmet']))),
            'count': frame_stats.get('violations', 0),
        })


# ---------------------------------------------------------------------------
# Core processing loop (shared by webcam + uploaded video)
# ---------------------------------------------------------------------------
def run_detection_loop(cap, frame_delay=None):
    frame_placeholder = st.empty()
    chip_placeholder = st.empty()
    alert_placeholder = st.empty()
    audio_placeholder = st.empty()

    st.session_state.session_start = st.session_state.session_start or time.time()
    last_time = time.time()

    try:
        while st.session_state.processing:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                st.session_state.processing = False
                break

            annotated_frame, frame_stats = st.session_state.detector.process_frame(frame)

            # FPS - smoothed over a rolling window
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                st.session_state.fps_window.append(1.0 / dt)
            fps = (sum(st.session_state.fps_window) / len(st.session_state.fps_window)) if st.session_state.fps_window else 0.0

            st.session_state.frame_count += 1
            if st.session_state.frame_count % 5 == 0:
                st.session_state.stats = st.session_state.detector.get_statistics()

            # Compliance trend sample every ~15 frames
            if st.session_state.frame_count % 15 == 0:
                stats = st.session_state.stats
                total = max(1, stats['total_frames'])
                compliance_pct = ((total - stats['violations']) / total) * 100
                elapsed = now - st.session_state.session_start
                st.session_state.compliance_history.append((round(elapsed, 1), round(compliance_pct, 1)))

            if frame_stats.get('new_violation_event'):
                log_violation_snapshot(annotated_frame, frame_stats)

            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            with frame_placeholder.container():
                st.markdown('<div class="sg-video-frame">', unsafe_allow_html=True)
                st.image(frame_rgb, channels="RGB", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            render_status_chips(fps, frame_stats)
            with chip_placeholder.container():
                pass  # chips already rendered above frame; placeholder kept for future use

            render_alert_banner(frame_stats, alert_placeholder, audio_placeholder)

            if frame_delay:
                elapsed_loop = time.time() - loop_start
                sleep_time = frame_delay - elapsed_loop
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        cap.release()


def process_webcam():
    cap = cv2.VideoCapture(st.session_state.camera_index)
    if not cap.isOpened():
        st.error("Could not access webcam. Check camera permissions or try a different camera index.")
        st.session_state.processing = False
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_CONFIG['display_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_CONFIG['display_height'])
    cap.set(cv2.CAP_PROP_FPS, 30)
    run_detection_loop(cap, frame_delay=None)


def process_video_file(uploaded_file):
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        st.error("Could not open video file.")
        os.remove(temp_path)
        st.session_state.processing = False
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / video_fps

    try:
        run_detection_loop(cap, frame_delay=frame_delay)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------------------------------------------------------------------
# Tabs: Analytics / Violation Log / Settings
# ---------------------------------------------------------------------------
def stat_card(col, icon, label, value, color=None, progress_pct=None, progress_color=None):
    with col:
        color_style = f"color:{color};" if color else ""
        html = f"""
        <div class="sg-card">
            <div class="sg-card-label">{icon} {label}</div>
            <div class="sg-card-value" style="{color_style}">{value}</div>
        """
        if progress_pct is not None:
            pc = progress_color or T['accent']
            html += f"""<div class="sg-progress-track"><div class="sg-progress-fill" style="width:{min(progress_pct,100)}%; background:{pc};"></div></div>"""
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)


def render_analytics_tab():
    stats = st.session_state.stats
    total = stats['total_frames']
    compliance_rate = ((total - stats['violations']) / total * 100) if total else 100.0
    detection_rate = (stats['total_detections'] / total * 100) if total else 0.0

    st.markdown('<div class="sg-section">Session Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    stat_card(c1, "📊", "Frames Processed", f"{total}")
    stat_card(c2, "🎯", "Total Detections", f"{stats['total_detections']}")
    stat_card(c3, "⚠️", "Violation Frames", f"{stats['violations']}", color=T['danger'],
              progress_pct=(stats['violations'] / total * 100) if total else 0, progress_color=T['danger'])
    stat_card(c4, "✅", "Compliance Rate", f"{compliance_rate:.1f}%", color=T['success'],
              progress_pct=compliance_rate, progress_color=T['success'])
    stat_card(c5, "📈", "Detection Rate", f"{detection_rate:.1f}%", color=T['accent'],
              progress_pct=min(detection_rate, 100), progress_color=T['accent'])

    st.markdown('<div class="sg-section" style="margin-top:1.2rem;">Compliance Trend</div>', unsafe_allow_html=True)
    if st.session_state.compliance_history:
        chart_data = {
            "Elapsed (s)": [p[0] for p in st.session_state.compliance_history],
            "Compliance %": [p[1] for p in st.session_state.compliance_history],
        }
        st.line_chart(chart_data, x="Elapsed (s)", y="Compliance %", height=260)
    else:
        st.markdown(
            f"<div class='sg-info'><div class='sg-info-title'>No data yet</div>"
            f"<div class='sg-info-details'>Start detection to build a live compliance trend.</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sg-section" style="margin-top:1.2rem;">System Status</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    model_ok = st.session_state.detector is not None
    stat_card(s1, "🤖", "AI Model", "Loaded" if model_ok else "Not loaded",
              color=T['success'] if model_ok else T['danger_soft'])
    stat_card(s2, "🔴" if st.session_state.processing else "⚪", "Detection",
              "Active" if st.session_state.processing else "Idle",
              color=T['danger'] if st.session_state.processing else T['text_muted'])
    stat_card(s3, "🔊", "Sound Alerts", "On" if st.session_state.sound_alert else "Off",
              color=T['accent'] if st.session_state.sound_alert else T['text_muted'])


def render_violation_log_tab():
    st.markdown('<div class="sg-section">Captured Violations</div>', unsafe_allow_html=True)
    log = list(st.session_state.violation_log)

    if not log:
        st.markdown(
            "<div class='sg-info'><div class='sg-info-title'>No violations logged yet</div>"
            "<div class='sg-info-details'>Confirmed incidents (after debounce) will appear here with a snapshot and timestamp.</div></div>",
            unsafe_allow_html=True,
        )
        return

    if st.button("🗑️ Clear Log", use_container_width=False):
        st.session_state.violation_log.clear()
        st.rerun()

    cols = st.columns(3)
    for idx, entry in enumerate(log):
        with cols[idx % 3]:
            if os.path.exists(entry['path']):
                st.image(entry['path'], use_container_width=True)
            st.markdown(
                f"<div class='sg-pill'>🕒 {entry['timestamp']}</div>"
                f"<div class='sg-pill' style='border-color:{T['danger']}; color:{T['danger_soft']};'>"
                f"missing: {', '.join(entry['missing_gear'])}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)


def render_settings_tab():
    st.markdown('<div class="sg-section">Required PPE</div>', unsafe_allow_html=True)
    st.caption("A worker is flagged as a violation if ANY of the checked items below aren't detected on them. "
               "Applies next time you click Start.")
    all_classes = list(dict.fromkeys(REQUIRED_CLASSES_DEFAULT + OPTIONAL_CLASSES_DEFAULT))
    st.session_state.required_classes = st.multiselect(
        "Required gear",
        options=all_classes,
        default=[c for c in st.session_state.required_classes if c in all_classes] or list(REQUIRED_CLASSES_DEFAULT),
        help="Detected-but-unchecked items (e.g. goggles, mask) are still shown on screen, just not counted as violations.",
    )

    st.markdown('<div class="sg-section" style="margin-top:1.2rem;">Detection Sensitivity</div>', unsafe_allow_html=True)
    st.session_state.confidence = st.slider(
        "PPE gear confidence threshold", 0.05, 0.90, float(st.session_state.confidence), 0.05,
        help="Lower catches more gear items (more false positives). Applies next time you Start.",
    )
    st.session_state.person_confidence = st.slider(
        "Person confidence threshold", 0.05, 0.90, float(st.session_state.person_confidence), 0.05,
        help="Lower catches more workers (more false positives). Applies next time you Start.",
    )
    st.session_state.debounce_frames = st.slider(
        "Alert debounce (frames)", 1, 15, int(st.session_state.debounce_frames), 1,
        help="How many consecutive violating frames are needed before an alert fires. Higher = fewer false alarms, slightly slower response.",
    )

    st.markdown('<div class="sg-section" style="margin-top:1.2rem;">Accuracy (crowded / group scenes)</div>', unsafe_allow_html=True)
    res_options = {"640 - fastest": 640, "960 - balanced": 960, "1280 - most accurate, slowest": 1280}
    res_labels = list(res_options.keys())
    current_res_label = next((k for k, v in res_options.items() if v == st.session_state.detection_resolution), res_labels[0])
    chosen_label = st.selectbox(
        "Detection resolution", res_labels, index=res_labels.index(current_res_label),
        help="Higher resolution helps the model see small or distant workers/gear in group shots, at the cost of speed. Applies next time you Start.",
    )
    st.session_state.detection_resolution = res_options[chosen_label]
    st.session_state.match_threshold = st.slider(
        "Gear-to-worker matching strictness", 0.10, 0.60, float(st.session_state.match_threshold), 0.05,
        help="How much of a detected gear item must overlap a worker to count as theirs. Lower helps when workers stand close together or gear is partly hidden; too low can attribute gear to the wrong nearby person. Applies next time you Start.",
    )

    st.session_state.sound_alert = st.toggle("🔊 Sound alert on new violation", value=st.session_state.sound_alert)
    st.session_state.camera_index = st.number_input(
        "Webcam device index", min_value=0, max_value=10, value=int(st.session_state.camera_index), step=1,
        help="Change this if you have multiple cameras and the wrong one opens.",
    )

    st.markdown('<div class="sg-section" style="margin-top:1.2rem;">About</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sg-card">
        <div class="sg-card-label">SafetyGuard AI</div>
        <div style="color:{T['text_muted']}; font-size:0.85rem; line-height:1.6; margin-top:0.4rem;">
        Real-time PPE compliance monitoring built on two YOLOv8n models: a stock
        person detector and a pretrained PPE-item detector (helmet, vest, gloves,
        goggles, mask, safety_shoe). Each item is matched to the worker it overlaps
        most, so violations are per-person, not per-frame. An alert only fires once a violation
        persists across several frames, reducing false alarms from a single
        missed detection.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    initialize_session_state()

    header_col, button_col = st.columns([5, 1], vertical_alignment="center")
    with header_col:
        st.markdown('<div class="sg-header">Safety Compliance Monitoring</div>', unsafe_allow_html=True)
        st.markdown('<div class="sg-subtitle">Real-time PPE detection &amp; safety compliance monitoring</div>', unsafe_allow_html=True)
    with button_col:
        if st.button("Load Model", type="primary", use_container_width=True):
            load_model()
        model_ok = st.session_state.detector is not None
        dot_color = T['success'] if model_ok else T['text_muted']
        st.markdown(
            f"<div style='text-align:right; margin-top:4px;'>"
            f"<span class='sg-pill'><span class='sg-dot' style='background:{dot_color}'></span> "
            f"{'Model ready' if model_ok else 'No model loaded'}</span></div>",
            unsafe_allow_html=True,
        )

    tab_live, tab_analytics, tab_log, tab_settings = st.tabs(
        ["🎥 Live Feed", "📊 Analytics", "📸 Violation Log", "⚙️ Settings"]
    )

    with tab_live:
        left, right = st.columns([1, 2.3], gap="medium")

        with left:
            st.markdown('<div class="sg-card">', unsafe_allow_html=True)
            st.markdown('<div class="sg-card-label">📹 Input Source</div>', unsafe_allow_html=True)
            source_type = st.radio(
                "Select Input", ["📷 Webcam", "🎬 Video File"], label_visibility="collapsed",
            )
            if source_type == "🎬 Video File":
                uploaded_file = st.file_uploader(
                    "Upload Video", type=['mp4', 'avi', 'mov', 'mkv'], label_visibility="collapsed",
                )
                st.session_state.video_source = uploaded_file
                if uploaded_file:
                    size_mb = uploaded_file.size / (1024 * 1024)
                    st.markdown(
                        f"<span class='sg-pill'>📎 {uploaded_file.name} ({size_mb:.1f}MB)</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.session_state.video_source = "webcam"
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="sg-card" style="margin-top:0.8rem;">', unsafe_allow_html=True)
            st.markdown('<div class="sg-card-label">🎮 Action Buttons</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("▶️ Start", type="primary", disabled=st.session_state.detector is None, use_container_width=True):
                    st.session_state.processing = True
                    st.session_state.frame_count = 0
                    st.session_state.session_start = time.time()
                    st.session_state.fps_window.clear()
                    if st.session_state.detector:
                        d = st.session_state.detector
                        d.ppe_cfg['confidence_threshold'] = st.session_state.confidence
                        d.ppe_cfg['imgsz'] = st.session_state.detection_resolution
                        d.person_cfg['confidence_threshold'] = st.session_state.person_confidence
                        d.person_cfg['imgsz'] = st.session_state.detection_resolution
                        d.alert_config['violation_persistence_frames'] = st.session_state.debounce_frames
                        d.required_classes = {c.lower() for c in st.session_state.required_classes}
                        d.match_cfg['containment_threshold'] = st.session_state.match_threshold
            with c2:
                if st.button("⏹️ Stop", type="secondary", use_container_width=True):
                    st.session_state.processing = False

            if st.button("🔄 Reset Stats", use_container_width=True):
                if st.session_state.detector:
                    st.session_state.detector.reset_statistics()
                    st.session_state.stats = st.session_state.detector.get_statistics()
                    st.session_state.frame_count = 0
                    st.session_state.compliance_history.clear()
                    st.session_state.session_start = time.time()
                    st.success("Stats reset")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:0.8rem;">
                <span class='sg-pill'>Frames: {st.session_state.stats['total_frames']}</span>
                <span class='sg-pill' style='color:{T['danger_soft']}; border-color:{T['danger']};'>
                    Violations: {st.session_state.stats['violations']}
                </span>
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown('<div class="sg-card-label" style="margin-bottom:0.4rem;">🎬 Video Player</div>', unsafe_allow_html=True)
            if st.session_state.processing:
                if st.session_state.video_source == "webcam":
                    process_webcam()
                elif st.session_state.video_source:
                    process_video_file(st.session_state.video_source)
                else:
                    st.session_state.processing = False
            else:
                st.markdown("""
                <div class="sg-info">
                    <div class="sg-info-title">Ready to start</div>
                    <div class="sg-info-details">Load a model, choose a source on the left, then click "Start".</div>
                </div>
                """, unsafe_allow_html=True)

    with tab_analytics:
        render_analytics_tab()

    with tab_log:
        render_violation_log_tab()

    with tab_settings:
        render_settings_tab()


if __name__ == "__main__":
    main()
