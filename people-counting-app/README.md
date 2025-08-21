# People Counting for Crowd Monitoring

An AI app that detects and counts people in images & videos. Built with **Ultralytics YOLOv8**, **Gradio**, and deployable on **Hugging Face Spaces (Docker)**.

## Features
- Bounding boxes on detected people + total count overlay
- Image & video tabs
- Per-frame CSV export for videos (`frame_index, timestamp_s, person_count`)
- CPU-friendly defaults (YOLOv8n, stride configurable)

## Local Run
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860)

## Docker Run

```bash
docker build -t people-counting .
docker run -p 7860:7860 people-counting
```

## Hugging Face Spaces (Docker)

1. Create a new Space → **Docker** runtime.
2. Upload `app.py`, `utils.py`, `requirements.txt`, `Dockerfile`, `README.md`.
3. Spaces will build the image and start the app automatically.

## Settings

* `MODEL_NAME` (env): `yolov8n.pt` (default) or `yolov8s.pt` for better accuracy
* `FRAME_STRIDE` (env): process every Nth frame (default 3)
* `CONF_THRESH`, `IOU_THRESH`: detection thresholds

## Notes

* CPU inference: prefer shorter/smaller videos, or increase `FRAME_STRIDE`.
* Class filter: only **person** (COCO class id 0) is enabled for speed.
* If you need zone-based counting or tracking, integrate **ByteTrack** or **SORT** and add polygon ROIs.

## License

MIT

