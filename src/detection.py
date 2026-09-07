"""
Real-time PPE Detection Pipeline using OpenCV and two YOLOv8n models.
Multi-class edition (no GPU / training required)
---------------------------------------------------------------------------
Two pretrained models run every frame:

  person_model - stock COCO YOLOv8n, filtered to class 0 ("person"), finds
                  where each worker is.

  ppe_model     - a YOLOv8n already fine-tuned to detect PPE items
                  (helmet, vest, gloves, goggles, mask, safety_shoe).

Each detected PPE item is matched to the person box it belongs to, so
compliance is evaluated PER WORKER: "this specific person is missing a
helmet", not just "no helmet visible anywhere in the frame". Alerts are
debounced across a few consecutive frames to avoid single-frame flicker.

v2 accuracy fixes (crowded / multi-worker scenes):
  - Item -> person matching now uses containment ratio (how much of the
    item sits inside the person box) AND, when a gear item is ambiguous
    between two overlapping people, breaks the tie by nearest box-center
    distance instead of silently going to whichever person happened first.
  - Inference resolution (imgsz) is configurable - raising it helps recall
    on small/distant workers in group shots, at some speed cost.
  - On-screen violation labels no longer stack on top of each other when
    people stand close together: each label is placed in the first free
    vertical slot above its box.
---------------------------------------------------------------------------
"""

import cv2
import numpy as np
import time


class PPEDetector:
    def __init__(self, person_model, ppe_model, config, draw_colors=None):
        """
        Args:
            person_model: a loaded ultralytics YOLO model (COCO weights)
            ppe_model: a loaded ultralytics YOLO model (PPE weights), or
                       None if it couldn't be downloaded/loaded - the
                       detector then falls back to person-only mode.
            config: dict with PERSON_MODEL_CONFIG, PPE_MODEL_CONFIG,
                    REQUIRED_CLASSES (list of lowercase class names),
                    ALERT_CONFIG, PPE_CLASS_COLORS_BGR, MATCH_CONFIG (optional)
            draw_colors: optional overrides for person-box colors
        """
        self.person_model = person_model
        self.ppe_model = ppe_model

        self.person_cfg = config['PERSON_MODEL_CONFIG']
        self.ppe_cfg = config['PPE_MODEL_CONFIG']
        self.alert_config = config.get('ALERT_CONFIG', {'violation_persistence_frames': 3})
        self.class_colors = config.get('PPE_CLASS_COLORS_BGR', {})
        self.match_cfg = config.get('MATCH_CONFIG', {'containment_threshold': 0.25})

        # Required classes are matched case-insensitively against whatever
        # names the PPE model reports.
        self.required_classes = {c.lower() for c in config.get('REQUIRED_CLASSES', [])}

        self.person_colors = draw_colors or {
            'ok': (140, 196, 46),        # BGR - person has all required gear
            'violation': (58, 57, 230),  # BGR - person missing required gear
            'unknown': (200, 200, 200),  # BGR - PPE model unavailable
        }

        self.ppe_class_names = {}
        if self.ppe_model is not None and hasattr(self.ppe_model, 'names'):
            self.ppe_class_names = {i: n.lower() for i, n in self.ppe_model.names.items()}

        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'violations': 0,
            'compliant_detections': 0,
        }

        self._violation_streak = 0
        self._alert_active = False
        self._label_rects = []   # reset every frame - used to avoid overlapping text labels

    # ------------------------------------------------------------------
    def process_frame(self, frame):
        """
        Returns:
            annotated_frame, frame_stats
        """
        t0 = time.time()
        self._label_rects = []

        persons = self._detect_persons(frame)
        items = self._detect_items(frame) if self.ppe_model is not None else []

        infer_ms = (time.time() - t0) * 1000.0

        frame_stats = {
            'people_detected': len(persons),
            'items_detected': len(items),
            'violations': 0,
            'compliant': 0,
            'missing_gear': [],
            'alert_active': False,
            'new_violation_event': False,
            'infer_ms': infer_ms,
            'ppe_model_available': self.ppe_model is not None,
        }

        annotated_frame = frame.copy()

        if self.ppe_model is None:
            # No PPE model - just show detected people, no compliance check.
            for p in persons:
                self._draw_box(annotated_frame, p, 'person', self.person_colors['unknown'])
            frame_stats['compliant'] = len(persons)
        else:
            assignments, unassigned_items = self._assign_items_to_persons(persons, items)

            missing_all = []
            for idx, person in enumerate(persons):
                matched_classes = {it[5].lower() for it in assignments[idx]}
                missing = self.required_classes - matched_classes if self.required_classes else set()

                color = self.person_colors['violation'] if missing else self.person_colors['ok']
                label = 'missing: ' + ', '.join(sorted(missing)) if missing else 'compliant'
                self._draw_box(annotated_frame, person, label, color)

                if missing:
                    frame_stats['violations'] += 1
                    missing_all.extend(missing)
                else:
                    frame_stats['compliant'] += 1

                for item in assignments[idx]:
                    self._draw_item(annotated_frame, item)

            for item in unassigned_items:
                self._draw_item(annotated_frame, item)

            frame_stats['missing_gear'] = missing_all

        # ---- Debounce: only flip to "alert" after N consecutive violating frames
        persistence = self.alert_config.get('violation_persistence_frames', 3)
        if frame_stats['violations'] > 0:
            self._violation_streak += 1
        else:
            self._violation_streak = 0

        was_active = self._alert_active
        self._alert_active = self._violation_streak >= persistence
        frame_stats['alert_active'] = self._alert_active
        frame_stats['new_violation_event'] = self._alert_active and not was_active

        # ---- Update running totals
        self.stats['total_frames'] += 1
        self.stats['total_detections'] += len(persons) + len(items)
        if frame_stats['alert_active']:
            self.stats['violations'] += 1
        else:
            self.stats['compliant_detections'] += 1

        return annotated_frame, frame_stats

    # ------------------------------------------------------------------
    # Model calls
    # ------------------------------------------------------------------
    def _detect_persons(self, frame):
        results = self.person_model(
            frame,
            conf=self.person_cfg['confidence_threshold'],
            iou=self.person_cfg.get('iou_threshold', 0.45),
            classes=[0],   # COCO class 0 = person
            verbose=False,
            imgsz=self.person_cfg.get('imgsz', 640),
            device='cpu',
        )
        boxes = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                boxes.append((x1, y1, x2, y2, conf, 'person'))
        return boxes

    def _detect_items(self, frame):
        results = self.ppe_model(
            frame,
            conf=self.ppe_cfg['confidence_threshold'],
            iou=self.ppe_cfg.get('iou_threshold', 0.45),
            verbose=False,
            imgsz=self.ppe_cfg.get('imgsz', 640),
            device='cpu',
        )
        boxes = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                name = self.ppe_class_names.get(class_id, f'class_{class_id}')
                boxes.append((x1, y1, x2, y2, conf, name))
        return boxes

    # ------------------------------------------------------------------
    # Item <-> person association
    # ------------------------------------------------------------------
    def _assign_items_to_persons(self, persons, items):
        """
        Assigns each PPE item to the person it belongs to.

        Step 1: compute a containment ratio (how much of the item box falls
        inside each person box). A helmet/glove is always tiny relative to a
        full-body box, so this - not symmetric IoU - is what correctly
        matches small items to a big person box.

        Step 2: in a crowd, people boxes often overlap each other, so an
        item can legitimately clear the containment threshold for more than
        one person. Ties (or near-ties) are broken by picking whichever
        person's box CENTER is closest to the item - the item almost always
        belongs to the nearest body, not a neighbor standing next to them.

        Items with no person clearing the threshold are left unassigned
        (still drawn, just not counted toward anyone's compliance).
        """
        threshold = self.match_cfg.get('containment_threshold', 0.25)
        assignments = [[] for _ in persons]
        unassigned = []

        person_centers = [((p[0] + p[2]) / 2.0, (p[1] + p[3]) / 2.0) for p in persons]

        for item in items:
            ix1, iy1, ix2, iy2 = item[0], item[1], item[2], item[3]
            icx, icy = (ix1 + ix2) / 2.0, (iy1 + iy2) / 2.0

            candidates = []  # (idx, ratio, distance)
            for idx, person in enumerate(persons):
                px1, py1, px2, py2 = person[0], person[1], person[2], person[3]
                ratio = self._containment_ratio((ix1, iy1, ix2, iy2), (px1, py1, px2, py2))
                if ratio >= threshold:
                    pcx, pcy = person_centers[idx]
                    dist = ((icx - pcx) ** 2 + (icy - pcy) ** 2) ** 0.5
                    candidates.append((idx, ratio, dist))

            if not candidates:
                unassigned.append(item)
                continue

            # Prefer strong containment first (rounded, so near-ties compete
            # on distance instead of noise), then nearest center.
            candidates.sort(key=lambda c: (-round(c[1], 2), c[2]))
            best_idx = candidates[0][0]
            assignments[best_idx].append(item)

        return assignments, unassigned

    @staticmethod
    def _containment_ratio(item_box, person_box):
        """Fraction of item_box's area that falls inside person_box."""
        ix1, iy1, ix2, iy2 = item_box
        px1, py1, px2, py2 = person_box
        ox1, oy1 = max(ix1, px1), max(iy1, py1)
        ox2, oy2 = min(ix2, px2), min(iy2, py2)
        ow, oh = max(0.0, ox2 - ox1), max(0.0, oy2 - oy1)
        overlap = ow * oh
        item_area = max(1e-6, (ix2 - ix1) * (iy2 - iy1))
        return overlap / item_area

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _draw_box(self, frame, entry, label, color):
        x1, y1, x2, y2, confidence, _ = entry
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        text = f"{label} ({confidence:.2f})"
        self._draw_label(frame, int(x1), int(y1), text, color)

    def _draw_item(self, frame, item):
        x1, y1, x2, y2, confidence, name = item
        color = self.class_colors.get(name, (255, 196, 94))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        text = f"{name}: {confidence:.2f}"
        cv2.putText(frame, text, (int(x1), int(y2) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    def _draw_label(self, frame, x, y, text, color):
        """
        Draw a filled text label anchored above (x, y), nudging it downward
        in fixed steps if it would overlap a label already placed this
        frame - keeps "missing: ..." tags legible when workers stand close
        together instead of stacking illegibly on top of each other.
        """
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        pad = 6
        box_h = th + 10
        rect_x1, rect_x2 = x, x + tw + pad
        rect_y2 = y

        for _ in range(6):  # cap attempts so we never loop forever
            rect_y1 = rect_y2 - box_h
            collision = any(
                rect_x1 < ox2 and rect_x2 > ox1 and rect_y1 < oy2 and rect_y2 > oy1
                for (ox1, oy1, ox2, oy2) in self._label_rects
            )
            if not collision:
                break
            rect_y2 += box_h + 4  # try a slot further down, into the frame

        self._label_rects.append((rect_x1, rect_y1, rect_x2, rect_y2))
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
        cv2.putText(frame, text, (rect_x1 + 3, rect_y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2)

    # ------------------------------------------------------------------
    def get_statistics(self):
        return self.stats

    def reset_statistics(self):
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'violations': 0,
            'compliant_detections': 0,
        }
        self._violation_streak = 0
        self._alert_active = False


class VideoProcessor:
    """Optional CLI helper for processing a video file outside Streamlit."""

    def __init__(self, detector, video_config):
        self.detector = detector
        self.video_config = video_config

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_config['display_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_config['display_height'])

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame, _ = self.detector.process_frame(frame)
            cv2.imshow('PPE Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
