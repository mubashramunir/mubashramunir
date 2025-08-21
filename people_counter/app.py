from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.pcounter import PeopleDetector

st.set_page_config(page_title="People Counter", page_icon="🧑‍🤝‍🧑", layout="wide")


def load_image_from_upload(uploaded_file) -> np.ndarray | None:
	if uploaded_file is None:
		return None
	data = uploaded_file.read()
	file_bytes = np.frombuffer(data, np.uint8)
	bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
	return bgr


@st.cache_resource

def get_detector() -> PeopleDetector:
	return PeopleDetector()


def main():
	st.title("People Counter 🧑‍🤝‍🧑")
	st.write("Upload an image or choose a video to count people using YOLOv8.")

	detector = get_detector()

	mode = st.sidebar.radio("Mode", ["Image", "Video"], horizontal=True)
	conf_threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)

	if mode == "Image":
		uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
		if uploaded_image is not None:
			bgr = load_image_from_upload(uploaded_image)
			if bgr is not None:
				result = detector.count_people_in_image(bgr, conf_threshold=conf_threshold)
				annotated = detector.annotate_image(bgr, result.detected_boxes_xyxy, result.confidences)
				st.metric("People count", result.person_count)
				st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Count: {result.person_count} | Inference {result.inference_ms:.1f} ms", use_column_width=True)

	else:
		video_file = st.file_uploader("Upload a short video (mp4, mov, avi)", type=["mp4", "mov", "avi"])
		if video_file is not None:
			tmp_path = Path("outputs/uploaded_video.mp4")
			tmp_path.parent.mkdir(parents=True, exist_ok=True)
			with open(tmp_path, "wb") as f:
				f.write(video_file.read())

			output_path = Path("outputs/annotated_streamlit.mp4")
			with st.spinner("Processing video..."):
				detector.process_video(str(tmp_path), str(output_path), conf_threshold=conf_threshold, max_frames=500)

			st.video(str(output_path))


if __name__ == "__main__":
	main()