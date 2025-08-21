import os
import cv2
import time
import math
import tempfile
import numpy as np
import gradio as gr
import pandas as pd
from ultralytics import YOLO
from utils import draw_boxes_and_count, yolo_results_to_lists

# ------------ Config ------------
MODEL_NAME = os.environ.get("MODEL_NAME", "yolov8n.pt")  # or yolov8s.pt
FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", 3))      # process every Nth frame
CONF_THRESH = float(os.environ.get("CONF_THRESH", 0.25))
IOU_THRESH = float(os.environ.get("IOU_THRESH", 0.45))

# Load model once (CPU)
model = YOLO(MODEL_NAME)
# Restrict to person class (id=0 in COCO)
PERSON_CLASS_ID = 0


def analyze_image(image: np.ndarray, conf: float = CONF_THRESH):
    if image is None:
        return None, 0
    results = model.predict(source=image, conf=conf, iou=IOU_THRESH, classes=[PERSON_CLASS_ID], verbose=False)
    boxes, labels, confs = yolo_results_to_lists(results)
    annotated, count = draw_boxes_and_count(image.copy(), boxes, labels, confs, show_conf=False)
    return annotated[:, :, ::-1], count  # BGR->RGB for Gradio


def analyze_video(video_path: str, stride: int = FRAME_STRIDE, conf: float = CONF_THRESH):
    if not video_path:
        return None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    # Prepare output video (mp4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp_out.name
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # CSV log
    rows = []

    idx = 0
    processed = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Timestamp in seconds
        ts = idx / fps

        if idx % stride == 0:
            results = model.predict(source=frame, conf=conf, iou=IOU_THRESH, classes=[PERSON_CLASS_ID], verbose=False)
            boxes, labels, confs = yolo_results_to_lists(results)
            frame, count = draw_boxes_and_count(frame, boxes, labels, confs, show_conf=False)
            processed += 1
        # If skipped frame, keep last count (simple hold), optional interpolation could be added
        rows.append({"frame_index": idx, "timestamp_s": round(ts, 3), "person_count": count if idx % stride == 0 else rows[-1]["person_count"] if rows else 0})

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()

    fps_eff = processed / max(1e-6, (time.time() - t0))
    print(f"Processed {processed} frames (stride={stride}) in {time.time() - t0:.2f}s — ~{fps_eff:.2f} FPS")

    # Save CSV
    df = pd.DataFrame(rows)
    csv_path = out_path.replace(".mp4", ".csv")
    df.to_csv(csv_path, index=False)

    return out_path, csv_path


with gr.Blocks(title="People Counting for Crowd Monitoring") as demo:
    gr.Markdown("""
    # 👥 People Counting for Crowd Monitoring
    Upload an **image** or **video** to detect people. The app draws bounding boxes and shows the total **count**.
    """)

    with gr.Tab("Image"):
        with gr.Row():
            in_img = gr.Image(type="numpy", label="Upload Image")
            with gr.Column():
                conf_img = gr.Slider(0.1, 0.9, value=CONF_THRESH, step=0.05, label="Confidence Threshold")
                btn_img = gr.Button("Analyze Image", variant="primary")
        out_img = gr.Image(type="numpy", label="Output", interactive=False)
        out_count = gr.Number(label="People Count", interactive=False, precision=0)
        btn_img.click(fn=analyze_image, inputs=[in_img, conf_img], outputs=[out_img, out_count])

    with gr.Tab("Video"):
        with gr.Row():
            in_vid = gr.Video(label="Upload Video")
            with gr.Column():
                conf_vid = gr.Slider(0.1, 0.9, value=CONF_THRESH, step=0.05, label="Confidence Threshold")
                stride_vid = gr.Slider(1, 10, value=FRAME_STRIDE, step=1, label="Frame Stride (process every N frames)")
                btn_vid = gr.Button("Analyze Video", variant="primary")
        out_vid = gr.Video(label="Annotated Video")
        out_csv = gr.File(label="Download per-frame CSV")
        btn_vid.click(fn=analyze_video, inputs=[in_vid, stride_vid, conf_vid], outputs=[out_vid, out_csv])

    gr.Markdown("Made with ❤️ using Ultralytics YOLOv8 (CPU).")

if __name__ == "__main__":
    # Spaces sets PORT; default to 7860
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

