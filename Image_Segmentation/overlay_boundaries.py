import argparse
from typing import Tuple
import numpy as np
from PIL import Image
import cv2


def parse_color(s: str) -> Tuple[int, int, int]:
    parts = s.split(',')
    if len(parts) != 3:
        raise ValueError('Color must be R,G,B')
    return tuple(int(p) for p in parts)


def load_mask(mask_path: str) -> np.ndarray:
    img = Image.open(mask_path).convert('L')
    arr = np.array(img)
    # binarize: positive if >0
    bw = (arr > 0).astype(np.uint8) * 255
    return bw


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert('RGB')


def find_contours_opencv(mask: np.ndarray):
    # mask: uint8, 0/255
    # OpenCV findContours expects single-channel image
    img = mask.copy()
    contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    # contours is a list of arrays of shape (N,1,2)
    pycontours = [c.reshape(-1, 2) for c in contours]
    return pycontours


def find_contours_skimage(mask: np.ndarray):
    # find_contours returns coords in (row, col) floats along boundary; convert to int
    contours = measure.find_contours(mask.astype(np.uint8), level=0.5)
    pycontours = []
    for c in contours:
        # c is array of (row, col) floats; convert to (x,y) ints
        pts = np.fliplr(c)  # to (x,y)
        pts = np.round(pts).astype(np.int32)
        pycontours.append(pts)
    return pycontours

def dilate_mask(mask: np.ndarray, dilate_px: int):
    if dilate_px <= 0:
        return mask

    kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)

def draw_contours_on_image(img: Image.Image, contours, color: Tuple[int, int, int], thickness: int = 2):

    arr = np.array(img)[:, :, ::-1].copy()  # RGB->BGR
    cv2_contours = [c.reshape(-1, 1, 2).astype(np.int32) for c in contours if len(c) > 0]
    cv2.drawContours(arr, cv2_contours, contourIdx=-1, color=(int(color[2]), int(color[1]), int(color[0])), thickness=thickness)
    return Image.fromarray(arr[:, :, ::-1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--out_path', required=True)
    parser.add_argument('--color', default='255,0,0')
    parser.add_argument('--thickness', type=int, default=2)
    parser.add_argument('--dilate', type=int, default=0)
    args = parser.parse_args()

    mask = load_mask(args.mask)
    if args.dilate > 0:
        mask = dilate_mask(mask, args.dilate)

    img = load_image(args.image)
    # check size match
    if mask.shape[1] != img.width or mask.shape[0] != img.height:
        # If sizes differ, try to resize mask to image
        mask = np.array(Image.fromarray(mask).resize((img.width, img.height), resample=Image.NEAREST))

    contours = find_contours_opencv(mask)

    color = parse_color(args.color)
    out_img = draw_contours_on_image(img, contours, color=color, thickness=args.thickness)
    out_img.save(args.out_path)
    print(f"Saved overlay image to {args.out_path}")


if __name__ == '__main__':
    main()
