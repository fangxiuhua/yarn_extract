"""Annular ROI detection via contour hierarchy analysis."""

import cv2 as cv
import numpy as np
from yarn import Yarn


def detect_yarn(binary: np.ndarray) -> Yarn | None:
    """Detect the annular (ring-shaped) yarn region from a binary image.

    Finds the largest outer contour and its largest child contour,
    fits enclosing circles, and builds an annular mask.

    Args:
        binary: Binary image (uint8, values 0 or 255).

    Returns:
        Yarn object if detection succeeds, None otherwise.
    """
    contours, hierarchy = cv.findContours(
        binary, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        print("No contours found in binary image")
        return None

    hierarchy = hierarchy[0]

    # --- Find the largest outer (parent) contour ---
    outer_idx = -1
    max_outer_area = 0
    for i, cnt in enumerate(contours):
        if hierarchy[i][3] == -1:  # no parent → outer contour
            area = cv.contourArea(cnt)
            if area > max_outer_area:
                max_outer_area = area
                outer_idx = i

    if outer_idx == -1:
        print("No outer contour found")
        return None

    # --- Find the largest inner (child) contour of that outer ---
    inner_idx = -1
    max_inner_area = 0
    for i, cnt in enumerate(contours):
        if hierarchy[i][3] == outer_idx:  # parent is the outer contour
            area = cv.contourArea(cnt)
            if area > max_inner_area:
                max_inner_area = area
                inner_idx = i

    # --- Fit outer circle ---
    (cx_outer, cy_outer), radius_outer = cv.minEnclosingCircle(
        contours[outer_idx]
    )

    # --- Fit inner circle (only if found) ---
    if inner_idx != -1:
        (cx_inner, cy_inner), radius_inner = cv.minEnclosingCircle(
            contours[inner_idx]
        )
    else:
        cx_inner, cy_inner = cx_outer, cy_outer
        radius_inner = 0

    # --- Build annular mask: fill outer, erase inner ---
    mask = np.zeros(binary.shape, dtype=np.uint8)
    cv.circle(mask, (int(cx_outer), int(cy_outer)), int(radius_outer), 255, -1)
    if inner_idx != -1:
        cv.circle(mask, (int(cx_inner), int(cy_inner)), int(radius_inner), 0, -1)

    return Yarn(
        center_in=(int(cx_inner), int(cy_inner)),
        center_out=(int(cx_outer), int(cy_outer)),
        radius_inner=int(radius_inner),
        radius_outer=int(radius_outer),
        mask=mask,
    )
