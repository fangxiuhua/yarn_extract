"""Image preprocessing: resize, grayscale, denoise, binarize."""

import cv2 as cv
import numpy as np


def preprocess(
    image: np.ndarray,
    max_dimension: int = 640,
    gaussian_kernel: tuple[int, int] = (5, 5),
    threshold: int = 127,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess a BGR image for contour detection.

    Steps: resize → grayscale → Gaussian blur → binary threshold.

    Args:
        image: Input BGR image.
        max_dimension: The larger edge is scaled to this size.
        gaussian_kernel: Kernel size for GaussianBlur.
        threshold: Binary threshold value (0-255).

    Returns:
        (gray_image, binary_image) tuple.
    """
    h, w = image.shape[:2]
    scale = max_dimension / max(h, w)
    resized = cv.resize(image, (int(w * scale), int(h * scale)))

    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, gaussian_kernel, 0)
    _, binary = cv.threshold(blurred, threshold, 255, cv.THRESH_BINARY)

    return gray, binary
