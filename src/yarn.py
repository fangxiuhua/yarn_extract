"""Yarn region-of-interest: extraction and polar unwrapping."""

import math

import cv2 as cv
import numpy as np


class Yarn:
    """Annular ROI detected on a yarn cone image.

    Holds inner/outer circle parameters and provides methods to
    extract the ring region and unwrap it to a rectangular image.

    Attributes:
        center_in: (x, y) of the inner circle center (tube edge).
        center_out: (x, y) of the outer circle center (cone edge).
        radius_inner: Inner circle radius in pixels.
        radius_outer: Outer circle radius in pixels.
        mask: Binary mask of the annular region.
    """

    def __init__(
        self,
        center_in: tuple[int, int],
        center_out: tuple[int, int],
        radius_inner: int,
        radius_outer: int,
        mask: np.ndarray,
    ) -> None:
        if radius_outer <= radius_inner:
            raise ValueError(
                f"radius_outer ({radius_outer}) must be > radius_inner ({radius_inner})"
            )
        self.center_in = center_in
        self.center_out = center_out
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        self.mask = mask

    def extract(self, img: np.ndarray) -> np.ndarray:
        """Extract the annular ROI from an image using the mask."""
        return cv.bitwise_and(img, img, mask=self.mask)

    def circle_to_rectangle(self, img: np.ndarray) -> np.ndarray:
        """Unwrap the annular region into a rectangle via polar mapping.

        Maps each pixel (row, col) in the output rectangle to a
        (rho, theta) polar coordinate around center_out, then
        samples the corresponding Cartesian pixel.

        Args:
            img: Grayscale image matching the mask dimensions.

        Returns:
            Rectangular image (height = ring width, width = outer circumference).
        """
        center_x, center_y = self.center_out
        ring_width = self.radius_outer - self.radius_inner

        output_width = int(2 * math.pi * self.radius_outer)
        output_height = ring_width

        rect_img = np.zeros((output_height, output_width), dtype=np.uint8)

        for row in range(output_height):
            rho = self.radius_outer - row - 1
            for col in range(output_width):
                theta = 2 * math.pi * col / output_width
                x = int(center_x + rho * math.cos(theta) + 0.5)
                y = int(center_y - rho * math.sin(theta) + 0.5)
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    rect_img[row, col] = img[y, x]

        return rect_img

    def draw(self, img: np.ndarray, thickness: int = 2) -> np.ndarray:
        """Draw the detected inner and outer circles on a copy of the image."""
        vis = img.copy()
        cv.circle(vis, self.center_out, self.radius_outer, (0, 255, 0), thickness)
        cv.circle(vis, self.center_in, self.radius_inner, (0, 255, 0), thickness)
        cv.circle(vis, self.center_out, 10, (0, 0, 255), -1)
        return vis
