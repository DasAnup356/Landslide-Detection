import argparse 
import joblib
from PIL import Image
import numpy as np

def predict(image_in:str, 
            model_in:str, 
            mask_out:str,
            target_patches=3500,
            proba_thresh = 0.6,
            remove_small_param = 100):
    #
    payload = joblib.load(model_in) #scikit-learn model
    clf = payload['model']          #has a standard function for scikit-learn model
    
    #
    img = Image.open(image_in).convert('RGB')
    W, H = img.size

    from utils.patch_operations import compute_square_patch_size
    s = compute_square_patch_size(W, 
                                  H, 
                                  target_patches=target_patches)
    print(f"Image size: {W}x{H}, chosen square patch size: {s} px")

    #
    from utils.patch_operations import pad_image
    padded, orig_w, orig_h = pad_image(img, 
                                       s)
    PW, PH = padded.size
    nx = PW //s
    ny = PH //s
    total_patches = nx * ny 
    print(f"Splitting into  {nx}x{ny} = {total_patches} square patches")

    #
    feats_list = []
    coords = []
    from utils.features import extract_features_from_patch
    for j in range(ny):
        for i in range(nx):
            left = i*s
            top = j*s 
            patch = padded.crop((left, top, left+s, top+s))

            feats, _ = extract_features_from_patch(patch)

            feats_list.append(feats)
            coords.append((left, top))
    
    #
    X = np.vstack(feats_list)
    print(f"Computed features for {X.shape[0]} patches (feature dim {X.shape[1]})")

    #predict
    #proba_thresh
    proba = clf.predict(X)

    mask = np.zeros((PH, PW), dtype=np.uint8)
    for (left, top), p in zip(coords, proba):
        if p >= proba_thresh:
            mask[top:top + s, left:left + s] = 255

    #crop back to original image
    mask = mask[:orig_h, :orig_w]

    #remove small objects 
    from skimage.morphology import remove_small_objects
    
    filter_size = remove_small_param
    try: 
        bool_mask = mask.astype(bool)
        cleaned = remove_small_objects(bool_mask, min_size= filter_size)
        mask = (cleaned*255).astype(np.uint8)
    except Exception as e: 
        print("removing small objects failed:", e)

    Image.fromarray(mask).save(mask_out)
    print(f"Saved mask to {mask_out}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_in', required=True)
    parser.add_argument('--model_in', required=True)
    parser.add_argument('--mask_out', required=True)
    parser.add_argument('--target_patches', type=int, default=3500)
    parser.add_argument('--proba_thresh', type=float, default=0.5)
    parser.add_argument('--remove_small_param', type=int, default=50)
    args = parser.parse_args()

    predict(args.image_in, 
            args.model_in,
            args.mask_out,
            target_patches=args.target_patches,
            proba_thresh=args.proba_thresh,
            remove_small_param=args.remove_small_param)
    
if __name__ == '__main__':
    main()