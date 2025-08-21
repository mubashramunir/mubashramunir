import cv2
import numpy as np
from typing import List, Tuple


def draw_boxes_and_count(frame: np.ndarray, boxes: List[Tuple[int,int,int,int]],
                         labels: List[str], confidences: List[float],
                         show_conf: bool = False) -> tuple[np.ndarray, int]:
    """Draw boxes & return (annotated_frame, count)."""
    count = 0
    for (x1, y1, x2, y2), label, conf in zip(boxes, labels, confidences):
        if label != "person":
            continue
        count += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = label if not show_conf else f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Count badge
    badge = f"Count: {count}"
    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    pad = 10
    cv2.rectangle(frame, (10, 10), (10 + tw + 2 * pad, 10 + th + 2 * pad), (0, 0, 0), -1)
    cv2.putText(frame, badge, (10 + pad, 10 + th + pad), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return frame, count


def yolo_results_to_lists(results) -> tuple[list, list, list]:
    """Convert Ultralytics YOLO results (single image) to (boxes, labels, confidences)."""
    boxes, labels, confs = [], [], []
    if not results or len(results) == 0:
        return boxes, labels, confs
    r0 = results[0]
    if r0.boxes is None:
        return boxes, labels, confs

    xyxy = r0.boxes.xyxy.cpu().numpy().astype(int)
    conf = r0.boxes.conf.cpu().numpy()
    cls = r0.boxes.cls.cpu().numpy().astype(int)
    names = r0.names

    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        label = names[int(k)]
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
        labels.append(label)
        confs.append(float(c))
    return boxes, labels, confs

