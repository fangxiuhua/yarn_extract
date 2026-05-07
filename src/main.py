"""Main entry point: load images, run the detection pipeline, display results."""

import cv2 as cv
import matplotlib.pyplot as plt

from config import (
    BINARY_THRESHOLD,
    DEFAULT_IMAGES,
    GAUSSIAN_KERNEL,
    MAX_DIMENSION,
)
from detection import detect_yarn
from preprocessing import preprocess


def main() -> None:
    image_paths = DEFAULT_IMAGES

    for img_path in image_paths:
        # 1. Load
        img = cv.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        # 2. Preprocess
        gray, binary = preprocess(
            img,
            max_dimension=MAX_DIMENSION,
            gaussian_kernel=GAUSSIAN_KERNEL,
            threshold=BINARY_THRESHOLD,
        )

        # 3. Detect annular ROI
        yarn = detect_yarn(binary)
        if yarn is None:
            print(f"No yarn region found: {img_path}")
            continue

        # 4. Extract ROI and unwrap
        extracted = yarn.extract(gray)
        rect_img = yarn.circle_to_rectangle(extracted)

        # 5. Display
        plt.imshow(rect_img, cmap="gray")
        plt.title(img_path)
        plt.axis("off")
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()
