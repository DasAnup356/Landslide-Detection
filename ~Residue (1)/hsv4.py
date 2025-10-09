import cv2
import numpy as np


def area_open(mask, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255
    return clean_mask

def detect_and_highlight_veg(input_path: str, output_path: str, min_area: int = 100):
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {input_path}")

    median = cv2.medianBlur(image, 5)
    smooth = cv2.bilateralFilter(median, d=9, sigmaColor=75, sigmaSpace=75)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    hsv[...,2] = clahe.apply(hsv[...,2])
    v = hsv[...,2]
    valid_v = cv2.inRange(v, 30, 240) 

    hsv_ranges = [
        # (1) Bright/normal green
        (np.array([35, 8, 4], np.uint8), np.array([96, 255, 181], np.uint8)),
        # (2) Olive/Brownish green
        (np.array([15, 39, 66], np.uint8), np.array([30, 113, 173], np.uint8)),
        # (3) Pale green/grey-green (backgrounds)
        (np.array([11, 8, 82], np.uint8), np.array([114, 46, 120], np.uint8)),
    ]
    
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    
    for lower, upper in hsv_ranges:
        m = cv2.inRange(hsv, lower, upper)
        m = cv2.bitwise_and(m, m, mask=valid_v)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)

        m_kept = area_open(m, min_area)    
        combined = cv2.bitwise_or(combined, m_kept)

    big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, big_kernel, iterations=1)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    overlay = image.copy()
    cv2.drawContours(overlay, kept, -1, (0,255,0), thickness=cv2.FILLED)
    highlighted = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    cv2.drawContours(highlighted, kept, -1, (0,255,255), 2)
    cv2.imwrite(output_path, highlighted)
    print(f"Saved with wider HSV vegetation highlight to {output_path}")
    
    filtered_mask = np.zeros_like(combined)
    cv2.drawContours(filtered_mask, kept, -1, 255, thickness=cv2.FILLED)
    return filtered_mask



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vegetation detection via wide-range multi-HSV compositing with preprocessing")
    parser.add_argument("-i","--input", required=True, help="Input image path")
    parser.add_argument("-o","--output", default="veg_wide_multi_hsv.png", help="Output image path")
    parser.add_argument("-v", "--no_veg", default="no_vegetation.png", help="Path for vegetation-removed output")
    parser.add_argument("-m","--min_area", type=int, default=1000, help="Min contour area")
    args = parser.parse_args()
    detect_and_highlight_veg(args.input, args.output, args.no_veg, args.min_area)
