import numpy as np 
from PIL import Image

def rgb_to_hsv(arr: np.ndarray) -> np.ndarray: # Input : numpy arr(HxWx3) in uint8
    
    from matplotlib.colors import rgb_to_hsv 
    arr_float = arr.astype(np.float32) / 255.0 #matplotlib expects floats in [0,1]
    hsv = rgb_to_hsv(arr_float.reshape(-1, 3)).reshape(arr.shape)
    return hsv 


def excess_green(arr: np.ndarray) -> float:
    
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)

    eg = 2*g-r-b
    return float(np.mean(eg))

def lbp_hist(patch_gray: np.ndarray, 
             P=24, R=3, n_bins=26) -> np.ndarray: 
    
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(patch_gray, P, R, method='uniform') #bins: [0, P+1] (uniform patterns)

    (hist, _) = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

def glcm_features(patch_gray: np.ndarray,
                  distance=[1],
                  angle=[0]) -> list:
    
    
    from skimage.feature import graycomatrix
    from skimage.feature import graycoprops

    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    levels = 256
    patch_q = patch_gray.astype(np.uint8)
    glcm = graycomatrix(patch_q, distances=distance, angles=angle,
                       levels=levels, symmetric=True, normed=True)
    feats = []
    for p in props:
        try:
            v = graycoprops(glcm, p).mean()
            feats.append(float(v))
        except Exception:
            feats.append(float('nan'))
    return feats

def edge_density(patch_gray: np.ndarray) -> float: 
    #input must be a gray image
    arr = patch_gray.astype(np.float32) 
    gy, gx = np.gradient(arr) #Sobel Filter 
    mag = np.sqrt(gx*gx  + gy*gy)
    return float(np.mean(mag) / 255.0)

def extract_features_from_patch(patch: Image.Image, 
                                use_lbp=True,
                                use_glcm=True) -> tuple[np.ndarray, list[str]]:
    arr = np.array(patch.convert('RGB'))
    h, w, _ = arr.shape

    #Color stats RGB 
    feats = []
    names = []
    for c, ch in enumerate(['R', 'G', 'B']):
        ch_data = arr[:, :, c].astype(np.float32)
        
        feats.append(float(np.mean(ch_data)))
        names.append(f'mean_{ch}')

        feats.append(float(np.std(ch_data)))
        names.append(f'std_{ch}')

    #HSV stats 
    from utils.features import rgb_to_hsv
    hsv = rgb_to_hsv(arr)
    for i, ch in enumerate(['H', 'S', 'V']): 
        feats.append(np.mean(hsv[:, :, i]))
        names.append(f'mean_{ch}')
        feats.append(float(np.std(hsv[:, :, i])))
        names.append(f'std_{ch}')
    
    #Excess Green features
    from utils.features import excess_green
    feats.append(excess_green(arr))
    names.append('excess_green')

    #Edge density
    from utils.features import edge_density
    gray = (0.2989 * arr[:, :, 0] 
            + 0.5870 * arr[:, :, 1] 
            + 0.1140 * arr[:, :, 2]).astype(np.uint8)
    feats.append(edge_density(gray))
    names.append('edge_density')

    #LBP hist
    from utils.features import lbp_hist
    if use_lbp:
        lbp_h = lbp_hist(gray)
    else:
        lbp_h = np.zeros(26, dtype=np.float32)
    for i in range(len(lbp_h)):
        feats.append(float(lbp_h[i]))
        names.append(f'lbp_{i}')

    #GLCM features
    from utils.features import glcm_features
    if use_glcm:
        glcm_f = glcm_features(gray)
    else:
        glcm_f = [0.0] * 6
    for i, nm in enumerate(['contrast', 'dissimilarity',
                            'homogeneity', 'energy',
                            'correlation', 'ASM']):
        feats.append(float(glcm_f[i]))
        names.append(f'glcm_{nm}')

    return np.array(feats, dtype=np.float32), names
