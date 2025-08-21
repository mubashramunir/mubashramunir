# People Counter (YOLOv8 + Streamlit)

A simple Python app to count the number of people in images or videos using YOLOv8. Includes a Streamlit UI and a CLI.

## Features
- Image people counting with bounding boxes
- Video processing with annotated output
- Streamlit web UI
- CLI for batch usage

## Quickstart

### 1) Setup a virtual environment
```bash
cd /workspace/people_counter
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 2) Run the Streamlit app
```bash
streamlit run app.py
```

### 3) Use the CLI
- Image:
```bash
python cli.py image path/to/image.jpg --save outputs/annotated.jpg
```
- Video:
```bash
python cli.py video path/to/video.mp4 --out outputs/annotated.mp4
```
- Camera (may not work in headless environments):
```bash
python cli.py camera 0
```

## Notes
- The first run will download the YOLOv8 weights (e.g., `yolov8n.pt`).
- CPU inference is supported; GPU is used automatically if available.
- For long videos, consider using the `--max-frames` option for a quick check.

## License
MIT