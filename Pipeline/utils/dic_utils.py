import numpy as np

def window_fft_correlation(ref_win: np.ndarray, 
                    def_win: np.ndarray) -> tuple[np.ndarray, float]:
    
    #Normalized window values
    ref_mean, ref_std = np.mean(ref_win), np.std(ref_win)
    def_mean, def_std = np.mean(def_win), np.std(def_win)

    texture_threshold = 5.0 #If image is too low contrast,
                            #skip correlation computation
    if ref_std < texture_threshold or def_std < texture_threshold:
        return None, 0.0

    ref_win_norm = (ref_win - ref_mean) / (ref_std + 1e-8)
    def_win_norm = (def_win - def_mean) / (def_std + 1e-8)
    
    #get Correlation Matrix
    from scipy.signal import fftconvolve
    corr_matrix = fftconvolve(def_win_norm, ref_win_norm[::-1, ::-1], mode='same')
    correlation_quality_value = np.max(corr_matrix) / np.sqrt(np.sum(ref_win_norm**2) * np.sum(def_win_norm**2)
                                                 + 1e-8)
    return corr_matrix, correlation_quality_value


def gaussian_subpixel_refinement(corr_matrix: np.ndarray,
                                 max_idx: tuple) -> tuple:
    y_max, x_max = max_idx
    y_max_refined = y_max
    x_max_refined = x_max
    
    #x
    try: 
        x_values = corr_matrix[y_max, x_max-1:x_max+2] #x_max-1, x_max, x_max+1, 
                                                        #if 3 points are not available, 
                                                        #it will throw error and return max_idx

        coeffs_x = np.polyfit([-1, 0, 1], x_values, 2) #y = ax^2 + bx + c
        if coeffs_x[0] <0: #Ensuring it's a peak: a<0
            x_peak_offset = -coeffs_x[1] / (2*coeffs_x[0]) #approx
            if abs(x_peak_offset) <= 1.0: 
                x_max_refined = x_max + x_peak_offset 
    except: pass
    
    #y
    try:             
        y_values = corr_matrix[y_max-1:y_max+2, x_max]  #x_max-1, x_max, x_max+1, 
                                                        #if 3 points are not available (for boundary points), 
                                                        #it will throw error and return max_idx

        coeffs_y = np.polyfit([-1, 0, 1], y_values, 2) #y = ax^2 + bx + c
        if coeffs_y[0] < 0:  # Ensure it's a peak (a < 0)
            y_peak_offset = -coeffs_y[1] / (2 * coeffs_y[0])
            if abs(y_peak_offset) <= 1.0:
                y_max_refined = y_max + y_peak_offset     
    except: pass 
    
    return (y_max_refined, x_max_refined)

def filter(U:np.ndarray,
           V:np.ndarray, correlation_quality) -> np.ndarray: 
        #filter on correaltion quality and vector size
    mag = np.hypot(U,V)
    
    correlation_threshold = 0.3 
    quality_mask = correlation_quality > correlation_threshold 
    #Only select windows whose correlation_quality is greater than 0.3
    
    if np.sum(quality_mask) > 0: 
        valid_mag = mag[quality_mask]
        adap_threshold = np.percentile(valid_mag, 95) #gives the 95% percentile value 
                                                #threshold value for vector size
    else: 
        adap_threshold = np.percentile(mag, 95)

    mag_mask = (mag > adap_threshold)
    final_mask = np.logical_and(quality_mask, mag_mask)
    return final_mask