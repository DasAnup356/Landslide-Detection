import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt
import os 

def load_grayscale(path): 
    #load Grayscale
    img = np.array(Image.open(path).convert('L'), dtype=np.float32)

    #Removes noise
    from scipy.ndimage import gaussian_filter
    img_filtered = gaussian_filter(img, sigma=0.8)

    #Improves Contrast
    import cv2 
    img_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(
        img_filtered.astype(np.uint8)
    )

    img = img_clahe.astype(np.float32)
    if np.std(img) <30: #low contrast image
        img_min, img_max = np.percentile(img, (2, 98))
        img = np.clip((img - img_min)* 255 / (img_max - img_min),
                       0, 255) #Stretch the histogram
    return img        

def compute_displacement_vectors(ref_img: np.ndarray,
                                 def_img: np.ndarray, *,
                                 subset_size: int = 9,
                                 step: int = 2,
                                 verbose: bool = False):
    
    #Creates complication in fft convolve
    if ref_img.shape != def_img.shape:
        raise ValueError("Reference and deformed imges must have the same shape")
    
    h, w = ref_img.shape
    y_coords = np.arange(subset_size//2, h - subset_size//2, step) # y-axis represtents row 
    x_coords = np.arange(subset_size//2, w -subset_size//2, step) # x-axis represents columns
    X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')

    U = np.zeros_like(X, dtype=np.float32)
    V = np.zeros_like(Y, dtype=np.float32)
    correlation_quality = np.zeros_like(X, dtype=np.float32)

    for i, y in enumerate(y_coords):
        if verbose and (i+1)% (max(1, len(y_coords)//10)) == 0 or i==0: 
            print(f"  Progress: {i+1}/{len(y_coords)} rows")

        for j, x in enumerate(x_coords): 
            ref_win = ref_img[y - subset_size//2 : y + subset_size//2,
                              x - subset_size//2 : x + subset_size//2]
            def_win = def_img[y - subset_size//2 : y + subset_size//2,
                              x - subset_size//2 : x + subset_size//2]
            
            #Gives Correlation Matrix for the window
            from utils.dic_utils import window_fft_correlation
            corr_matrix, correlation_quality[i, j] = window_fft_correlation(ref_win, 
                                                                     def_win)
    
            if corr_matrix is None:
                continue 
            else: 
                #Makes subspixel accuracy correction
                max_idx = np.unravel_index(np.argmax(corr_matrix), 
                                           corr_matrix.shape) #max index in 2d for 2d matrix
                
                from utils.dic_utils import gaussian_subpixel_refinement
                refined_idx = gaussian_subpixel_refinement(corr_matrix, 
                                                           max_idx)
                
                #Get Displacement Vectors
                center = np.array(corr_matrix.shape) //2  #coordinates
                shift  = np.array(refined_idx) - center
                V[i, j] = shift[0]                 #Displacement Vector_y
                U[i, j] = shift[1]                 #Displacement Vector_x
            
    #filter
    from utils.dic_utils import filter 
    final_mask = filter(U, V, correlation_quality)
            
    return X, Y, U, V, final_mask, correlation_quality
            
def veg_segmentation(X: np.ndarray,
                     Y: np.ndarray,
                     U: np.ndarray,
                     V: np.ndarray,
                     veg_mask: np.ndarray): 
    if veg_mask is not None:
        # Ensure coordinates are within bounds
        valid_y = np.clip(Y.astype(int), 0, veg_mask.shape[0] - 1)
        valid_x = np.clip(X.astype(int), 0, veg_mask.shape[1] - 1)
        
        # Create mask for vegetation regions (white pixels = 255 in mask)
        mask_indices = (veg_mask[valid_y, valid_x] == 255)
        
        # Set displacement vectors to None where vegetation is detected
        U[mask_indices] = None
        V[mask_indices] = None
    return U, V

def compute_statistics(U: np.ndarray,
                       V: np.ndarray, 
                       mask: np.ndarray,
                       correlation_quality: np.ndarray = None):
    #None, NaN values create problems in calculation of stats
    U = np.where((U is None) | np.isnan(U), 0, U)
    V = np.where((V is None) | np.isnan(V), 0, V)
    
    mag = np.hypot(U, V)

    valid_U = U[mask]
    valid_V = V[mask]
    valid_mag = mag[mask]

    stats = {
        'avg_U': valid_U.mean() if len(valid_U) > 0 else 0,
        'avg_V': valid_V.mean() if len(valid_V) > 0 else 0,
        'std_U': valid_U.std() if len(valid_U) > 0 else 0,
        'std_V': valid_V.std() if len(valid_V) > 0 else 0,
        'avg_total': valid_mag.mean() if len(valid_mag) > 0 else 0,
        'mag_of_avg': np.hypot(valid_U.mean(), valid_V.mean()) if len(valid_U) > 0 else 0,
        'ratio_pos_V': np.sum(valid_V > 0) / valid_V.size if valid_V.size else 0,
        'count_pos_V': np.sum(valid_V > 0),
        'count_total': valid_V.size
    } 

    if correlation_quality is not None:
        valid_quality = correlation_quality[mask]
        if len(valid_quality) > 0: 
                        stats.update({
                'avg_correlation': valid_quality.mean(),
                'min_correlation': valid_quality.min(),
                'max_correlation': valid_quality.max(),
                'correlation_std': valid_quality.std()
            })

    return stats 

def format_statistics(stats: dict) -> str:
    stats_text = (
        f"Avg U: {stats['avg_U']:.2f} ± {stats['std_U']:.2f} px\n"
        f"Avg V: {stats['avg_V']:.2f} ± {stats['std_V']:.2f} px\n"
        f"Avg total: {stats['avg_total']:.2f} px\n"
        f"Mag of avg: {stats['mag_of_avg']:.2f} px\n"
        f"Ratio +V: {stats['ratio_pos_V']:.2f} "
        f"({stats['count_pos_V']}/{stats['count_total']})"
    )

    if 'avg_correlation' in stats:
        corr_text = (
            f"\nAvg correlation: {stats['avg_correlation']:.3f}\n"
            f"Min correlation: {stats['min_correlation']:.3f}\n"
            f"Max correlation: {stats['max_correlation']:.3f}"
        )
        stats_text += corr_text
    return stats_text

def plot_vectors_stats(ref_img: np.ndarray,
                       X: np.ndarray,
                       Y: np.ndarray, 
                       U: np.ndarray,
                       V: np.ndarray,
                       mask: np.ndarray,
                       stats_text: str,
                       title: str,
                       correlation_quality: np.ndarray = None,
                       save_plot_dir:str = None) -> None:
    
    fig = plt.figure(figsize=(16, 7), dpi=150)
    
    # Create subplot layout with space for colorbar
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.15)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])
    
    # Vectors plot
    ax1.imshow(ref_img, cmap='gray', aspect='equal')
    
    if np.sum(mask) > 0:
        ax1.quiver(X[mask], Y[mask], U[mask], V[mask],
                  color='red', angles='xy', scale_units='xy', scale=1.0,
                  width=0.003, alpha=0.85, headwidth=3.5, headlength=4.5)
    else:
        ax1.text(0.5, 0.5, 'No Significant Vectors', transform=ax1.transAxes,
                ha='center', va='center', fontsize=14, color='red', weight='bold')
    
    ax1.set_title(title, fontsize=14, pad=15)
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.tick_params(labelsize=10)
    
    # Correlation plot
    if correlation_quality is not None:
        im = ax2.imshow(correlation_quality, cmap='viridis', aspect='equal', vmin=0, vmax=1)
        ax2.set_title('Correlation Map', fontsize=14, pad=15)
        
        # Colorbar
        cbar = fig.colorbar(im, cax=cax, aspect=20)
        cbar.set_label('Correlation Quality')
        cbar.ax.tick_params(labelsize=10)
    else:
        ax2.text(0.5, 0.5, 'No Correlation Data', transform=ax2.transAxes,
                ha='center', va='center', fontsize=14, color='gray')
        ax2.set_title('Correlation Map', fontsize=14, fontweight='bold', pad=15)
        cax.axis('off')
    
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.tick_params(labelsize=10)

    fig.text(0.02, 0.96, stats_text, fontsize=10, va='top', ha='left',
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.85, edgecolor='navy'),
             family='monospace', transform=fig.transFigure)
    plt.subplots_adjust(top=0.85, left=0.08, right=0.95)

    if save_plot_dir is not None:
        plt.savefig(os.path.join(save_plot_dir, "displacement_analysis.png"),
                    dpi=300,
                    bbox_inches='tight')
        print(f"Results saved to {save_plot_dir}")
    
    plt.show()
    return None

def save_displacement_to_csv(X, Y, U, V, filename):
    import csv

    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f) 
        writer.writerow(['X', 'Y', 'U', 'V'])
        for i in range(X.shape[0]):
                writer.writerow([X[i],
                                 Y[i],
                                 U[i],
                                 V[i]])
    return None

def main(ref_path:str,
         def_path:str, *,
         veg_mask_path:str = None,
         save_plot_dir:str = None):

    if ref_path is None:
        raise ValueError(f"ref_path cannot be empty")
    
    if def_path is None:
        raise ValueError(f"def_path cannot be empty")
    
    if veg_mask_path is not None: 
        veg_mask = np.array(Image.open(veg_mask_path).convert('L'), 
                                dtype=np.float32)
    else: 
        veg_mask = None

    #Input
    ref_img = load_grayscale(ref_path)
    def_img = load_grayscale(def_path)

    #Compute_displacement_vectors
    subset_size = 25
    step = 5 
    X, Y, U, V, mask, correlation_quality = compute_displacement_vectors(ref_img,
                                                                        def_img,
                                                                        subset_size=subset_size,
                                                                        step=step,
                                                                        verbose=True)
    # #Veg Segmentation 
    U, V = veg_segmentation(X, Y, U, V, veg_mask)

    #Plot
    stats = compute_statistics(U, V, mask)
    stats_text = format_statistics(stats)

    for key, value in stats.items(): 
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else: 
            print(f"{key}: {value}")

    plot_vectors_stats(ref_img, X, Y, U, V, mask, stats_text,
                       "Reference Image with Displacement Vectors",
                       correlation_quality,
                       save_plot_dir=save_plot_dir)
    
    #save displacement to csv
    save_displacement_to_csv(X[mask], Y[mask], U[mask], V[mask],
                              './output/displacement_vectors.csv')
    return 0 

if __name__ == "__main__":
    from PIL import Image

    ref_path = None
    def_path = None
    veg_mask_path = None 
    save_plot_dir = None #"./output"

    main(ref_path=ref_path, 
         def_path=def_path, 
         veg_mask_path=veg_mask_path,
         save_plot_dir=save_plot_dir)
