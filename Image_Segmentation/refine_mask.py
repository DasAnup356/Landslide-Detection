import cv2
import numpy as np
from skimage import measure
import argparse

def refine_mask(mask_path, image_path, mask_out, overlay_out):
    # Load mask (binary: vegetation=255, background=0) - vegetation is marked as white 
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    #reference image
    image = cv2.imread(image_path)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    H, W = image.shape[:2]
    min_area = (H * W) / 3500 *2

    #HSV range for Vegetation 
    H_low, H_high = 35, 85
    S_min = 50

    #Label connected components
    labeled_mask = measure.label(mask > 0, connectivity=2)
    refined_mask = np.zeros_like(mask)

    for region in measure.regionprops(labeled_mask):
        if region.area == 0:
            continue

        # Bounding box
        minr, minc, maxr, maxc = region.bbox
        blob_mask = (labeled_mask[minr:maxr, minc:maxc] == region.label)

        # HSV of blob
        blob_hsv = image_hsv[minr:maxr, minc:maxc][blob_mask]
        mean_h, mean_s, mean_v = np.mean(blob_hsv, axis=0)

        # Filter: discard small + not in HSV range
        if region.area < min_area and not (H_low <= mean_h <= H_high and mean_s > S_min):
            continue
        else:
            refined_mask[minr:maxr, minc:maxc][blob_mask] = 255

    # Save refined mask
    cv2.imwrite(mask_out, refined_mask)

    # Create overlay (boundaries on reference image)
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = image.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)  # green boundaries

    cv2.imwrite(overlay_out, overlay)

    print(f"Refined mask saved to {mask_out}")
    print(f"Overlay saved to {overlay_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=str, required=True, help="Path to input mask.png")
    parser.add_argument("--image", type=str, required=True, help="Path to reference/original image")
    parser.add_argument("--mask_out_path", type=str, default="./ouptut/refined_mask.png", help="Path to save refined mask")
    parser.add_argument("--overlay_out_path", type=str, default="./output/overlay.png", help="Path to save overlay image")
    args = parser.parse_args()

    refine_mask(args.mask, args.image, args.mask_out_path, args.overlay_out_path)
