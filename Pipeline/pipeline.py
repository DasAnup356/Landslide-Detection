import numpy as np
import argparse
from PIL import Image

from rf_predict import predict
from refine_mask import refine_mask

import tempfile
import os

def main():
    parser = argparse.ArgumentParser(description='Vegetation segmentation and displacement analysis pipeline')
    parser.add_argument("--ref_image", type=str, required=True, help="Path to reference image")
    parser.add_argument("--def_image", type=str, required=True, help="Path to deformed image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to vegetation segmentation model")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for results")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Create vegetation segmentation mask
    print("Creating vegetation segmentation mask...")
    temp_mask_path = os.path.join(tempfile.gettempdir(), "temp_veg_mask.png")
    predict(args.ref_image, args.model_path, temp_mask_path)
    
    # Step 2: Refine the vegetation mask
    print("Refining vegetation mask...")
    refined_mask_path = os.path.join(args.output_dir, "refined_veg_mask.png")
    overlay_path = os.path.join(args.output_dir, "overlay.png")
    refine_mask(temp_mask_path, args.ref_image, refined_mask_path, overlay_path)
    
    # Load the refined vegetation mask
    # veg_mask = cv2.imread(refined_mask_path, cv2.IMREAD_GRAYSCALE)

    from dic_fft import main 
    main(args.ref_image,
         args.def_image,
         veg_mask_path=refined_mask_path,
         save_plot_dir=args.output_dir)

if __name__ == "__main__":
    main()