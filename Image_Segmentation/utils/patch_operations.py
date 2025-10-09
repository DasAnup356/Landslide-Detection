import numpy as np 
import math
from PIL import Image

#Input image is assumed to be horizontal(landscape)
def compute_square_patch_size(W: int, 
                              H: int, 
                              target_patches: int = 3500) -> int:
    # patch_size(square) = s,
    # (W//s, H//s) ~ target_patches=3500
    if target_patches <= 0: 
        s = min(W,H) #length of a patch size 
        return s

    s = int( round(math.sqrt((W*H) / float(target_patches))) )
    s = max(8, s) #patch_size is at least 8 px
                  #to avoid extremely small patches (classifier won't work well)
    s = min(s, min(W, H)) #patch_size should be at maximum = min(W, H)

    #
    best_s = s 
    best_diff = abs( (math.ceil(W/s)* math.ceil(H/s)) 
                    - target_patches ) 
    for candidate in range( max(8, s-8), min(min(W,H), s+8) + 1):

        pcount = math.ceil(W / candidate) * math.ceil(H / candidate)
        d = abs(pcount - target_patches)
        if d < best_diff:
            best_diff = d
            best_s = candidate
    return best_s

def pad_image(img: Image.Image,
                          s: int):
    W, H = img.size
    new_w = math.ceil(W/s) * s
    new_h = math.ceil(H/s) * s
    if new_w == W and new_h == H: 
        return img, W, H 
    new_img = Image.new('RGB',
                        (new_w, new_h),
                        (0, 0, 0))  #creates new full black image
    new_img.paste(img, (0, 0)) #paste the original image from top-left corner (0,0)
    return new_img, W, H 

