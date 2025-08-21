import os
from pathlib import Path

import click
import cv2
from rich import print

from src.pcounter import PeopleDetector


@click.group()
def cli():
	"""People counting CLI using YOLOv8."""
	pass


@cli.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--conf", "conf_threshold", default=0.25, show_default=True, help="Confidence threshold")
@click.option("--save", "save_path", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Optional path to save annotated image")
def image(image_path: Path, conf_threshold: float, save_path: Path | None):
	"""Count people in an IMAGE and optionally save annotated copy."""
	detector = PeopleDetector()
	bgr = cv2.imread(str(image_path))
	if bgr is None:
		raise click.ClickException(f"Failed to read image: {image_path}")
	result = detector.count_people_in_image(bgr, conf_threshold=conf_threshold)
	annotated = detector.annotate_image(bgr, result.detected_boxes_xyxy, result.confidences)
	print(f"People count: [bold]{result.person_count}[/bold]  (inference: {result.inference_ms:.1f} ms)")
	if save_path:
		os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
		cv2.imwrite(str(save_path), annotated)
		print(f"Saved annotated image to [green]{save_path}[/green]")


@cli.command()
@click.argument("video_path", type=str)
@click.option("--conf", "conf_threshold", default=0.25, show_default=True, help="Confidence threshold")
@click.option("--out", "output_path", type=click.Path(dir_okay=False, path_type=Path), default="outputs/annotated.mp4", show_default=True)
@click.option("--max-frames", type=int, default=None, help="Process at most this many frames")
def video(video_path: str, conf_threshold: float, output_path: Path, max_frames: int | None):
	"""Count people in a VIDEO file; saves annotated output video."""
	detector = PeopleDetector()
	detector.process_video(video_path, str(output_path) if output_path else None, conf_threshold=conf_threshold, max_frames=max_frames)
	print(f"Annotated video saved to [green]{output_path}[/green]")


@cli.command()
@click.argument("camera_index", type=int, default=0)
@click.option("--conf", "conf_threshold", default=0.35, show_default=True, help="Confidence threshold")
@click.option("--out", "output_path", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--max-frames", type=int, default=None)
def camera(camera_index: int, conf_threshold: float, output_path: Path | None, max_frames: int | None):
	"""Open a camera stream by index (0 by default)."""
	detector = PeopleDetector()
	video_source = str(camera_index)
	detector.process_video(video_source, str(output_path) if output_path else None, conf_threshold=conf_threshold, max_frames=max_frames)


if __name__ == "__main__":
	cli()