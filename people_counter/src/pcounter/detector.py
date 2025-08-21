from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
	from ultralytics import YOLO  # Imported only when needed for model loading
except Exception:
	YOLO = None  # Allow import of this module even if ultralytics isn't installed yet


@dataclass
class DetectionResult:
	person_count: int
	detected_boxes_xyxy: List[Tuple[int, int, int, int]]
	confidences: List[float]
	inference_ms: float


class PeopleDetector:
	"""YOLOv8-based people detector that counts persons in images and video.

	This class loads a YOLOv8 model lazily and provides convenience methods to run
	detection, filter for the 'person' class, and annotate frames.
	"""

	def __init__(self, model_name: str = "yolov8n.pt", device: Optional[str] = None) -> None:
		self.model_name = model_name
		self.device = device
		self._model = None  # Lazy initialization

	def _get_model(self):
		"""Load and cache YOLO model on first use."""
		global YOLO
		if self._model is not None:
			return self._model
		if YOLO is None:
			from ultralytics import YOLO as _YOLO
			YOLO = _YOLO
		model = YOLO(self.model_name)
		if self.device:
			try:
				model.to(self.device)
			except Exception:
				pass
		self._model = model
		return self._model

	def count_people_in_image(self, bgr_image: np.ndarray, conf_threshold: float = 0.25) -> DetectionResult:
		"""Run detection on a single BGR image and return people count and boxes."""
		if bgr_image is None or bgr_image.size == 0:
			raise ValueError("Input image is empty or None")

		model = self._get_model()
		start_time = time.time()
		results = model.predict(source=bgr_image, verbose=False, conf=conf_threshold)
		elapsed_ms = (time.time() - start_time) * 1000.0

		person_class_id = 0
		boxes_xyxy: List[Tuple[int, int, int, int]] = []
		confidences: List[float] = []

		if not results:
			return DetectionResult(0, boxes_xyxy, confidences, elapsed_ms)

		r0 = results[0]
		if r0.boxes is None or len(r0.boxes) == 0:
			return DetectionResult(0, boxes_xyxy, confidences, elapsed_ms)

		for box, cls_id, conf in zip(r0.boxes.xyxy.cpu().numpy(), r0.boxes.cls.cpu().numpy(), r0.boxes.conf.cpu().numpy()):
			if int(cls_id) == person_class_id:
				x1, y1, x2, y2 = box.astype(int).tolist()
				boxes_xyxy.append((x1, y1, x2, y2))
				confidences.append(float(conf))

		return DetectionResult(len(boxes_xyxy), boxes_xyxy, confidences, elapsed_ms)

	def annotate_image(self, bgr_image: np.ndarray, boxes_xyxy: List[Tuple[int, int, int, int]], confidences: List[float]) -> np.ndarray:
		"""Draw bounding boxes and labels on the BGR image."""
		annotated = bgr_image.copy()
		for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confidences):
			cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
			label = f"person {conf:.2f}"
			cv2.putText(annotated, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
		return annotated

	def process_video(self, input_path: str, output_path: Optional[str] = None, conf_threshold: float = 0.25, max_frames: Optional[int] = None) -> None:
		"""Process a video file or camera stream; optionally write annotated output."""
		cap_source: Optional[object]
		try:
			cap_source = int(input_path)
		except (ValueError, TypeError):
			cap_source = input_path

		cap = cv2.VideoCapture(cap_source)
		if not cap.isOpened():
			raise FileNotFoundError(f"Could not open video source: {input_path}")

		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer: Optional[cv2.VideoWriter] = None

		if output_path:
			dirname = os.path.dirname(output_path) or "."
			os.makedirs(dirname, exist_ok=True)

		frame_count = 0
		while True:
			ret, frame = cap.read()
			if not ret:
				break

			result = self.count_people_in_image(frame, conf_threshold=conf_threshold)
			annotated = self.annotate_image(frame, result.detected_boxes_xyxy, result.confidences)
			cv2.putText(annotated, f"Count: {result.person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

			if output_path:
				if writer is None:
					h, w = annotated.shape[:2]
					fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
					writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
				writer.write(annotated)

			frame_count += 1
			if max_frames is not None and frame_count >= max_frames:
				break

		cap.release()
		if writer is not None:
			writer.release()