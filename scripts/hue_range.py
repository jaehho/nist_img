#!/usr/bin/env python3
"""
hue_selector.py

Usage:
    python hue_selector.py path/to/image.jpg

This script creates one window (“Hue Selector”) with two trackbars:
  • “Start Hue”   (0 to 179)
  • “Hue Length”  (1 to 180)

Press ESC or “q” to quit.
"""

import sys
import cv2
import numpy as np

def nothing(x):
    pass

def build_gradient_images(start_hue: int, hue_length: int, s_min=100, v_min=100):
    height, width = 200, 280

    # Build an HSV gradient
    h_vals = np.linspace(0, 179, width).astype(np.uint8)
    s_plane = np.full((height,), 255, dtype=np.uint8)
    v_plane = np.full((height,), 255, dtype=np.uint8)

    hsv_img = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        hsv_img[:, x, 0] = h_vals[x]
        hsv_img[:, x, 1] = s_plane
        hsv_img[:, x, 2] = v_plane

    # Convert to BGR for display in OpenCV
    grad_bgr = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    # Compute the wrap‐around hue interval
    end_hue = (start_hue + hue_length) % 180
    if hue_length <= 0:
        hue_length = 1
        end_hue = (start_hue + 1) % 180

    # Build a binary mask over the HSV gradient
    if start_hue < end_hue:
        lower = np.array([start_hue, s_min, v_min])
        upper = np.array([end_hue,   255,   255])
        mask = cv2.inRange(hsv_img, lower, upper)
    else:
        # wrap‐around case: [start_hue..179] ∪ [0..end_hue]
        lower1 = np.array([start_hue, s_min, v_min])
        upper1 = np.array([179,      255,   255])
        lower2 = np.array([0,        s_min, v_min])
        upper2 = np.array([end_hue,  255,   255])
        m1 = cv2.inRange(hsv_img, lower1, upper1)
        m2 = cv2.inRange(hsv_img, lower2, upper2)
        mask = cv2.bitwise_or(m1, m2)

    # Apply that mask to the BGR gradient
    masked_grad = cv2.bitwise_and(grad_bgr, grad_bgr, mask=mask)

    return grad_bgr, masked_grad

def build_frame_masked(frame_bgr: np.ndarray, frame_hsv: np.ndarray,
                       start_hue: int, hue_length: int):
    end_hue = (start_hue + hue_length) % 180
    if hue_length <= 0:
        hue_length = 1
        end_hue = (start_hue + 1) % 180

    if start_hue < end_hue:
        lower = np.array([start_hue, 100, 100])
        upper = np.array([end_hue,   255, 255])
        mask = cv2.inRange(frame_hsv, lower, upper)
    else:
        lower1 = np.array([start_hue, 100, 100])
        upper1 = np.array([179,      255, 255])
        lower2 = np.array([0,        100, 100])
        upper2 = np.array([end_hue,  255, 255])
        m1 = cv2.inRange(frame_hsv, lower1, upper1)
        m2 = cv2.inRange(frame_hsv, lower2, upper2)
        mask = cv2.bitwise_or(m1, m2)

    return cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

def main():
    if len(sys.argv) < 2:
        print("Usage: python hue_selector.py path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    frame_bgr = cv2.imread(img_path)
    if frame_bgr is None:
        print(f"Error: could not load image at '{img_path}'")
        sys.exit(1)

    # Precompute HSV version of the frame once
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Create a single named window for both trackbars and display
    window_name = "Hue Selector"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Create two trackbars inside that window
    cv2.createTrackbar("Start Hue",   window_name, 150, 179, nothing)
    cv2.createTrackbar("Hue Length",  window_name,  40, 180, nothing)

    while True:
        # Read current positions of the two trackbars
        start_hue  = cv2.getTrackbarPos("Start Hue", window_name)
        hue_length = cv2.getTrackbarPos("Hue Length", window_name)
        if hue_length < 1:
            hue_length = 1

        # Build the two small gradient images
        grad_bgr, masked_grad = build_gradient_images(start_hue, hue_length)

        # Build the masked version of the actual frame
        frame_masked = build_frame_masked(frame_bgr, frame_hsv, start_hue, hue_length)

        # Stack the two images vertically
        left_stack = np.vstack((grad_bgr, masked_grad))

        # Resize both masked‐frame and original frame to height
        h_grad_stack = left_stack.shape[0]
        fh, fw = frame_masked.shape[:2]
        if fh > 0:
            scale_mid = h_grad_stack / fh
            mid_w = int(fw * scale_mid)
            middle_resized = cv2.resize(frame_masked, (mid_w, h_grad_stack))
        else:
            middle_resized = frame_masked.copy()

        # Also resize the original image to height
        fh2, fw2 = frame_bgr.shape[:2]
        if fh2 > 0:
            scale_right = h_grad_stack / fh2
            right_w = int(fw2 * scale_right)
            right_resized = cv2.resize(frame_bgr, (right_w, h_grad_stack))
        else:
            right_resized = frame_bgr.copy()

        # Concatenate left, middle, and right horizontally
        combined = np.hstack((left_stack, middle_resized, right_resized))

        # Show the final combined image
        cv2.imshow(window_name, combined)

        # Wait 30 ms for keypress; if ESC or 'q', break out and quit
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
