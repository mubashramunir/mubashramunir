"""People counting package using YOLOv8.

Exports:
- PeopleDetector: Core detector class for images and video streams.
"""

from .detector import PeopleDetector

__all__ = ["PeopleDetector"]