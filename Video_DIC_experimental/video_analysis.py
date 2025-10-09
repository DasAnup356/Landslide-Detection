import numpy as np
import cv2
import math
import joblib
import os
import json
from pathlib import Path
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage import measure


class VegetationMasker:
    def __init__(self, model_path="rf_model.joblib"):
        try:
            self.model_payload = joblib.load(model_path)
            self.clf = self.model_payload['model']
        except Exception as e:
            print(f"Warning: Could not load vegetation model: {e}")
            self.clf = None
    #
    def rgb_to_hsv(self, arr: np.ndarray) -> np.ndarray:
        try:
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = hsv[:, :, 0] / 179.0
            hsv[:, :, 1] = hsv[:, :, 1] / 255.0
            hsv[:, :, 2] = hsv[:, :, 2] / 255.0
            return hsv
        except Exception:
            hsv = np.zeros_like(arr, dtype=np.float32)
            hsv[:, :, 2] = arr.max(axis=2) / 255.0
            return hsv
    #
    def excess_green(self, arr: np.ndarray) -> float:

        r = arr[:, :, 0].astype(np.float32)
        g = arr[:, :, 1].astype(np.float32)
        b = arr[:, :, 2].astype(np.float32)
        eg = 2 * g - r - b
        return float(np.mean(eg))
    #
    def lbp_hist(self, patch_gray: np.ndarray, P=24, R=3, n_bins=26) -> np.ndarray:
        try:
            lbp = local_binary_pattern(patch_gray, P, R, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            return hist
        except Exception:
            return np.zeros(n_bins, dtype=np.float32)

    #    
    def glcm_features(self, 
                      patch_gray: np.ndarray, 
                      distances=[1], 
                      angles=[0]) -> list:
        try:
            levels = 256
            patch_q = patch_gray.astype(np.uint8)
            glcm = graycomatrix(patch_q, distances=distances, angles=angles, 
                              levels=levels, symmetric=True, normed=True)
            feats = []
            props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
            for p in props:
                try:
                    v = graycoprops(glcm, p).mean()
                except Exception:
                    v = 0.0
                feats.append(float(v))
            return feats
        except Exception:
            return [0.0] * 6
        
    #    
    def edge_density(self, 
                     patch_gray: np.ndarray) -> float:
        arr = patch_gray.astype(np.float32)
        gy, gx = np.gradient(arr)
        mag = np.sqrt(gx * gx + gy * gy)
        return float(np.mean(mag) / 255.0)
    
    #
    def extract_features_from_patch(self,
                                    patch: Image.Image):
        arr = np.array(patch.convert('RGB'))
        feats = []
        for c in range(3):
            ch = arr[:, :, c].astype(np.float32)
            feats.append(float(np.mean(ch)))
            feats.append(float(np.std(ch)))
        hsv = self.rgb_to_hsv(arr)
        for i in range(3):
            feats.append(float(np.mean(hsv[:, :, i])))
            feats.append(float(np.std(hsv[:, :, i])))
        feats.append(self.excess_green(arr))
        gray = (0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]).astype(np.uint8)
        feats.append(self.edge_density(gray))
        lbp_h = self.lbp_hist(gray)
        feats.extend([float(x) for x in lbp_h])
        glcm_f = self.glcm_features(gray)
        feats.extend([float(x) for x in glcm_f])
        return np.array(feats, dtype=np.float32)
    
    #
    def compute_square_patch_size(self,
                                 W: int,
                                 H: int,
                                 target_patches: int = 3500) -> int:

        if target_patches <= 0:
            return max(1, min(W, H))
        s = int(round(math.sqrt((W * H) / float(target_patches))))
        s = max(8, s)
        s = min(s, min(W, H))
        best_s = s
        best_diff = abs((math.ceil(W / s) * math.ceil(H / s)) - target_patches)
        for candidate in range(max(8, s - 8), min(min(W, H), s + 8) + 1):
            pcount = math.ceil(W / candidate) * math.ceil(H / candidate)
            d = abs(pcount - target_patches)
            if d < best_diff:
                best_diff = d
                best_s = candidate
        return best_s
    
    def generate_vegetation_mask(self, image_path: str, target_patches=3500, proba_thresh=0.6):
        """Generate vegetation mask using RF model"""
        if self.clf is None:
            print("Warning: No vegetation model loaded. Creating dummy mask.")
            img = Image.open(image_path)
            return np.zeros((img.height, img.width), dtype=np.uint8)
        img = Image.open(image_path).convert('RGB')
        W, H = img.size
        s = self.compute_square_patch_size(W, H, target_patches=target_patches)
        new_w = math.ceil(W / s) * s
        new_h = math.ceil(H / s) * s
        if new_w != W or new_h != H:
            padded = Image.new('RGB', (new_w, new_h), (0, 0, 0))
            padded.paste(img, (0, 0))
        else:
            padded = img
        PW, PH = padded.size
        nx = PW // s
        ny = PH // s
        feats_list = []
        coords = []
        for j in range(ny):
            for i in range(nx):
                left = i * s
                top = j * s
                patch = padded.crop((left, top, left + s, top + s))
                feats = self.extract_features_from_patch(patch)
                feats_list.append(feats)
                coords.append((left, top))
        X = np.vstack(feats_list)
        if hasattr(self.clf, 'predict_proba'):
            proba = self.clf.predict_proba(X)[:, 1]
        else:
            proba = self.clf.predict(X).astype(float)
        mask = np.zeros((PH, PW), dtype=np.uint8)
        for (left, top), p in zip(coords, proba):
            if p >= proba_thresh:
                mask[top:top + s, left:left + s] = 255
        mask = mask[:H, :W]
        return mask
    
    def refine_mask_hsv(self, mask, image_path):
        """Refine vegetation mask using HSV analysis"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return mask
            
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            H, W = image.shape[:2]
            min_area = (H * W) / 3500 * 2
            H_low, H_high = 35, 85
            S_min = 50

            labeled_mask = measure.label(mask > 0, connectivity=2)
            refined_mask = np.zeros_like(mask)
            for region in measure.regionprops(labeled_mask):
                if region.area == 0:
                    continue
                minr, minc, maxr, maxc = region.bbox
                blob_mask = (labeled_mask[minr:maxr, minc:maxc] == region.label)
                blob_hsv = image_hsv[minr:maxr, minc:maxc][blob_mask]
                mean_h, mean_s, mean_v = np.mean(blob_hsv, axis=0)
                if region.area >= min_area or (H_low <= mean_h <= H_high and mean_s > S_min):
                    refined_mask[minr:maxr, minc:maxc][blob_mask] = 255
            return refined_mask
        
        except Exception as e:
            print(f"Warning: HSV refinement failed: {e}")
            return mask
        
class VideoRegionalAnalyzer:

    def __init__(self, model_path="rf_model.joblib", subset_size=16, step=4, region_size=50):
        self.subset_size = subset_size
        self.step = step
        self.region_size = region_size
        self.veg_masker = VegetationMasker(model_path)
        self.frames = []
        self.frame_times = []
        self.frame_pair_results = []
        self.temporal_differences = []

    def extract_frames_from_video(self, video_path, interval_seconds=10, output_dir="frames"):

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        frame_interval = int(fps * interval_seconds)
        extracted_frames = []
        frame_count = 0
        saved_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = frame_count / fps
                filename = f"frame_{saved_count:04d}_{timestamp:.1f}s.png"
                frame_path = output_path / filename
                Image.fromarray(frame_rgb).save(str(frame_path))
                extracted_frames.append({
                    'path': str(frame_path),
                    'timestamp': timestamp,
                    'frame_number': frame_count
                })
                self.frames.append(str(frame_path))
                self.frame_times.append(timestamp)
                print(f"Extracted frame {saved_count + 1}: t={timestamp:.1f}s")
                saved_count += 1
            frame_count += 1
        cap.release()
        return extracted_frames
    
    def load_image_grayscale(self, path: str) -> np.ndarray:

        img = np.array(Image.open(path).convert("L"), dtype=np.float32)
        img_filtered = gaussian_filter(img, sigma=0.8)
        img_uint8 = np.clip(img_filtered, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img = clahe.apply(img_uint8).astype(np.float32)
        return img

    def gaussian_subpixel_refinement(self, corr, max_idx):

        try:
            y_max, x_max = max_idx
            if (y_max <= 0 or y_max >= corr.shape[0]-1 or
                x_max <= 0 or x_max >= corr.shape[1]-1):
                return max_idx
            x_values = corr[y_max, x_max-1:x_max+2]
            if len(x_values) == 3:
                coeffs = np.polyfit([-1, 0, 1], x_values, 2)
                if coeffs[0] < 0:
                    x_offset = -coeffs[1] / (2 * coeffs[0])
                    if abs(x_offset) <= 1.0:
                        x_refined = x_max + x_offset
                    else:
                        x_refined = x_max
                else:
                    x_refined = x_max
            else:
                x_refined = x_max
            y_values = corr[y_max-1:y_max+2, x_max]
            if len(y_values) == 3:
                coeffs = np.polyfit([-1, 0, 1], y_values, 2)
                if coeffs[0] < 0:
                    y_offset = -coeffs[1] / (2 * coeffs[0])
                    if abs(y_offset) <= 1.0:
                        y_refined = y_max + y_offset
                    else:
                        y_refined = y_max
                else:
                    y_refined = y_max
            else:
                y_refined = y_max
            return (y_refined, x_refined)
        except:
            return max_idx
        
    def compute_displacement_field(self, ref_img, def_img):

        h, w = ref_img.shape
        y_coords = np.arange(self.subset_size//2, h - self.subset_size//2, self.step)
        x_coords = np.arange(self.subset_size//2, w - self.subset_size//2, self.step)
        Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
        U = np.zeros_like(Y, dtype=float)
        V = np.zeros_like(Y, dtype=float)
        correlation_quality = np.zeros_like(Y, dtype=float)
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                ref_win = ref_img[y - self.subset_size//2 : y + self.subset_size//2,
                                x - self.subset_size//2 : x + self.subset_size//2]
                def_win = def_img[y - self.subset_size//2 : y + self.subset_size//2,
                                x - self.subset_size//2 : x + self.subset_size//2]
                ref_mean, ref_std = np.mean(ref_win), np.std(ref_win)
                def_mean, def_std = np.mean(def_win), np.std(def_win)
                if ref_std < 5.0 or def_std < 5.0:
                    continue
                ref_win_norm = (ref_win - ref_mean) / ref_std
                def_win_norm = (def_win - def_mean) / def_std
                corr = fftconvolve(def_win_norm, ref_win_norm[::-1, ::-1], mode='same')
                max_idx = np.unravel_index(np.argmax(corr), corr.shape)
                max_corr = corr[max_idx]
                refined_idx = self.gaussian_subpixel_refinement(corr, max_idx)
                center = np.array(corr.shape) / 2
                shift = np.array(refined_idx) - center
                V[i, j] = shift[0]
                U[i, j] = shift[1]
                correlation_quality[i, j] = max_corr
        quality_mask = correlation_quality > 0.3
        magnitude = np.sqrt(U**2 + V**2)
        if np.sum(quality_mask) > 0:
            valid_magnitudes = magnitude[quality_mask]
            mag_threshold = np.percentile(valid_magnitudes, 95)
        else:
            mag_threshold = 10.0
        magnitude_mask = magnitude < mag_threshold
        final_mask = quality_mask & magnitude_mask
        return X, Y, U, V, final_mask, correlation_quality
    
    def filter_vectors_by_vegetation(self, X, Y, U, V, mask, veg_mask):

        filtered_mask = mask.copy()
        vegetation_removed_count = 0
        for i in range(Y.shape[0]):
            for j in range(X.shape[1]):
                if mask[i, j]:
                    pixel_y = int(round(Y[i, j]))
                    pixel_x = int(round(X[i, j]))
                    if (0 <= pixel_y < veg_mask.shape[0] and 
                        0 <= pixel_x < veg_mask.shape[1]):
                        if veg_mask[pixel_y, pixel_x] > 127:
                            filtered_mask[i, j] = False
                            vegetation_removed_count += 1
        return filtered_mask
    
    def compute_regional_displacement_sums(self, X, Y, U, V, mask):

        max_x = int(np.max(X)) + self.subset_size
        max_y = int(np.max(Y)) + self.subset_size
        n_regions_x = math.ceil(max_x / self.region_size)
        n_regions_y = math.ceil(max_y / self.region_size)
        regional_U_sum = np.zeros((n_regions_y, n_regions_x))
        regional_V_sum = np.zeros((n_regions_y, n_regions_x))
        regional_count = np.zeros((n_regions_y, n_regions_x))
        for i in range(Y.shape[0]):
            for j in range(X.shape[1]):
                if mask[i, j]:
                    vec_x = X[i, j]
                    vec_y = Y[i, j]
                    region_idx_x = int(vec_x // self.region_size)
                    region_idx_y = int(vec_y // self.region_size)
                    if (0 <= region_idx_y < n_regions_y and 
                        0 <= region_idx_x < n_regions_x):
                        regional_U_sum[region_idx_y, region_idx_x] += U[i, j]
                        regional_V_sum[region_idx_y, region_idx_x] += V[i, j]
                        regional_count[region_idx_y, region_idx_x] += 1
        regional_net_magnitude = np.sqrt(regional_U_sum**2 + regional_V_sum**2)
        valid_regions_mask = regional_count > 0
        results = {
            'region_size': self.region_size,
            'n_regions': (n_regions_y, n_regions_x),
            'U_sum': regional_U_sum,
            'V_sum': regional_V_sum,
            'count': regional_count,
            'net_magnitude': regional_net_magnitude,
            'valid_regions_mask': valid_regions_mask
        }
        return results
    
    def process_all_frame_pairs(self):

        if len(self.frames) < 2:
            raise ValueError("Need at least 2 frames for displacement computation")
        print(f"\nProcessing {len(self.frames)-1} frame pairs for regional displacement analysis...")

        for i in range(len(self.frames) - 1):
            ref_img = self.load_image_grayscale(self.frames[i])
            def_img = self.load_image_grayscale(self.frames[i+1])
            X, Y, U, V, mask, quality = self.compute_displacement_field(ref_img, def_img)
            veg_mask = self.veg_masker.generate_vegetation_mask(
                self.frames[i], target_patches=3500, proba_thresh=0.6
            )
            veg_mask_refined = self.veg_masker.refine_mask_hsv(veg_mask, self.frames[i])
            filtered_mask = self.filter_vectors_by_vegetation(X, Y, U, V, mask, veg_mask_refined)
            regional_results = self.compute_regional_displacement_sums(X, Y, U, V, filtered_mask)
            frame_pair_data = {
                'frame_pair_index': i,
                'frame_indices': (i, i+1),
                'time_interval': (self.frame_times[i], self.frame_times[i+1]),
                'displacement_field': {
                    'X': X, 'Y': Y, 'U': U, 'V': V, 
                    'mask_original': mask, 'mask_filtered': filtered_mask,
                    'correlation_quality': quality
                },
                'vegetation_mask': veg_mask_refined,
                'regional_results': regional_results
            }
            self.frame_pair_results.append(frame_pair_data)
            valid_regions = np.sum(regional_results['valid_regions_mask'])
            if valid_regions > 0:
                valid_net_mag = regional_results['net_magnitude'][regional_results['valid_regions_mask']]
                print(f"  Regional analysis: {valid_regions} valid regions")

    def compute_temporal_differences(self):

        if len(self.frame_pair_results) < 2:
            print("Warning: Need at least 2 frame pairs for temporal differencing")
            return
        for i in range(len(self.frame_pair_results) - 1):
            pair_current = self.frame_pair_results[i]['regional_results']
            pair_next = self.frame_pair_results[i+1]['regional_results']
            if (pair_current['n_regions'] != pair_next['n_regions'] or
                pair_current['region_size'] != pair_next['region_size']):
                print(f"Warning: Regional structure mismatch for pairs {i} and {i+1}")
                continue
            U_diff = pair_next['U_sum'] - pair_current['U_sum']
            V_diff = pair_next['V_sum'] - pair_current['V_sum']
            net_diff_magnitude = np.sqrt(U_diff**2 + V_diff**2)
            valid_mask = pair_current['valid_regions_mask'] & pair_next['valid_regions_mask']
            temporal_diff = {
                'difference_index': i,
                'frame_pair_indices': (i, i+1),
                'time_intervals': (
                    self.frame_pair_results[i]['time_interval'],
                    self.frame_pair_results[i+1]['time_interval']
                ),
                'region_size': pair_current['region_size'],
                'n_regions': pair_current['n_regions'],
                'U_diff': U_diff,
                'V_diff': V_diff,
                'net_diff_magnitude': net_diff_magnitude,
                'valid_regions_mask': valid_mask
            }
            self.temporal_differences.append(temporal_diff)
            if np.sum(valid_mask) > 0:
                valid_diff_mag = net_diff_magnitude[valid_mask]
        print(f"\nCompleted computing {len(self.temporal_differences)} temporal differences")

    def create_final_overlay_plot(self, output_dir="video_analysis"):

        if not self.temporal_differences:
            return
        ref_img = self.load_image_grayscale(self.frames[0])
        fig, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=150)
        ax.imshow(ref_img, cmap='gray', alpha=0.8)
        first_diff = self.temporal_differences[0]
        n_regions_y, n_regions_x = first_diff['n_regions']
        region_size = first_diff['region_size']
        region_centers_x = np.arange(n_regions_x) * region_size + region_size // 2
        region_centers_y = np.arange(n_regions_y) * region_size + region_size // 2
        Y_centers, X_centers = np.meshgrid(region_centers_y, region_centers_x, indexing='ij')
        all_magnitudes = []
        all_vectors = []
        for diff_idx, diff_data in enumerate(self.temporal_differences):
            valid_mask = diff_data['valid_regions_mask']
            if np.sum(valid_mask) > 0:
                U_diff = diff_data['U_diff'][valid_mask]
                V_diff = diff_data['V_diff'][valid_mask]
                X_pos = X_centers[valid_mask]
                Y_pos = Y_centers[valid_mask]
                magnitudes = diff_data['net_diff_magnitude'][valid_mask]
                all_magnitudes.extend(magnitudes)
                all_vectors.append({
                    'X': X_pos,
                    'Y': Y_pos,
                    'U': U_diff,
                    'V': V_diff,
                    'magnitudes': magnitudes,
                    'time_index': diff_idx
                })
        if not all_vectors:
            return
        max_magnitude = np.max(all_magnitudes) if all_magnitudes else 1.0
        scale_factor = region_size / max_magnitude * 0.8
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_vectors)))
        for vector_set, color in zip(all_vectors, colors):
            ax.quiver(
                vector_set['X'], vector_set['Y'],
                vector_set['U'], vector_set['V'],
                vector_set['magnitudes'],
                cmap='plasma',
                scale=scale_factor,
                scale_units='xy',
                angles='xy',
                width=0.004,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
        if all_magnitudes:
            sm = plt.cm.ScalarMappable(cmap='plasma', 
                                     norm=plt.Normalize(vmin=0, vmax=max_magnitude))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label('Region Cumulative Vectors Magnitude (pixels)', fontsize=12)
        for x in range(0, ref_img.shape[1], region_size):
            ax.axvline(x, color='cyan', alpha=0.3, linewidth=0.5)
        for y in range(0, ref_img.shape[0], region_size):
            ax.axhline(y, color='cyan', alpha=0.3, linewidth=0.5)
        ax.set_title(f'Temporal Difference Overlay on Reference Image\n'
                    f'{region_size}px Regions',
                    fontsize=14)
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        summary_text = (
            f"Video Duration: {self.frame_times[-1]:.1f}s\n"
            f"Frame Pairs: {len(self.frame_pair_results)}\n"
            f"Region Size: {region_size}x{region_size} px\n"
        )
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
        plt.tight_layout()
        plt.savefig(f'{output_dir}/final_temporal_overlay.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def visualize_temporal_analysis(self, output_dir="video_analysis"):

        os.makedirs(output_dir, exist_ok=True)
        if not self.temporal_differences:
            return
        n_pairs = len(self.frame_pair_results)
        n_diffs = len(self.temporal_differences)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        for idx, pair_idx in enumerate([0, n_pairs//2, n_pairs-1]):
            if pair_idx < len(self.frame_pair_results):
                pair_data = self.frame_pair_results[pair_idx]
                regional = pair_data['regional_results']
                ax = axes[0, idx]
                net_mag_display = regional['net_magnitude'].copy()
                net_mag_display[~regional['valid_regions_mask']] = np.nan
                im = ax.imshow(net_mag_display, cmap='hot', origin='upper')
                plt.colorbar(im, ax=ax, label='Net Displacement (px)')
                time_start = pair_data['time_interval'][0]
                time_end = pair_data['time_interval'][1]
                ax.set_title(f'Frame Pair {pair_idx+1}\nt={time_start:.1f}s → {time_end:.1f}s')
                ax.set_xlabel('Region X')
                ax.set_ylabel('Region Y')
        for idx, diff_idx in enumerate([0, n_diffs//2, n_diffs-1]):
            if diff_idx < len(self.temporal_differences):
                diff_data = self.temporal_differences[diff_idx]
                ax = axes[1, idx]
                diff_mag_display = diff_data['net_diff_magnitude'].copy()
                diff_mag_display[~diff_data['valid_regions_mask']] = np.nan
                im = ax.imshow(diff_mag_display, cmap='RdBu_r', origin='upper')
                plt.colorbar(im, ax=ax, label='Difference Magnitude (px)')
                ax.set_title(f'Temporal Difference {diff_idx+1}\nPair({diff_idx+2}) - Pair({diff_idx+1})')
                ax.set_xlabel('Region X')
                ax.set_ylabel('Region Y')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/temporal_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
        self.create_final_overlay_plot(output_dir)

    def save_results(self, output_dir="video_analysis"):

        os.makedirs(output_dir, exist_ok=True)
        metadata = {
            'processing_date': datetime.now().isoformat(),
            'num_frames': len(self.frames),
            'num_frame_pairs': len(self.frame_pair_results),
            'num_temporal_differences': len(self.temporal_differences),
            'frame_times': self.frame_times,
            'parameters': {
                'subset_size': self.subset_size,
                'step': self.step,
                'region_size': self.region_size
            }
        }
        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        for i, pair_data in enumerate(self.frame_pair_results):
            regional = pair_data['regional_results']
            np.savez_compressed(
                f'{output_dir}/frame_pair_{i:04d}.npz',
                U_sum=regional['U_sum'],
                V_sum=regional['V_sum'],
                count=regional['count'],
                net_magnitude=regional['net_magnitude'],
                valid_regions_mask=regional['valid_regions_mask']
            )
        for i, diff_data in enumerate(self.temporal_differences):
            np.savez_compressed(
                f'{output_dir}/temporal_diff_{i:04d}.npz',
                U_diff=diff_data['U_diff'],
                V_diff=diff_data['V_diff'],
                net_diff_magnitude=diff_data['net_diff_magnitude'],
                valid_regions_mask=diff_data['valid_regions_mask']
            )
        print(f"All results saved to {output_dir}/")

def analyze_video_temporal_displacement(video_path, model_path="rf_model.joblib", 
                                               interval_seconds=10, region_size=50,
                                               subset_size=16, step=4, 
                                               output_dir="video_analysis"):

    print(" VIDEO TEMPORAL DISPLACEMENT ANALYSIS")
    analyzer = VideoRegionalAnalyzer(model_path, subset_size, step, region_size)
    frames_dir = f"{output_dir}/frames"
    analyzer.extract_frames_from_video(video_path, interval_seconds, frames_dir)
    analyzer.process_all_frame_pairs()
    analyzer.compute_temporal_differences()
    analyzer.visualize_temporal_analysis(output_dir)
    analyzer.save_results(output_dir)
    print("\n TEMPORAL ANALYSIS SUMMARY:")
    if analyzer.temporal_differences:
        all_diff_magnitudes = []
        for diff_data in analyzer.temporal_differences:
            if np.sum(diff_data['valid_regions_mask']) > 0:
                valid_diffs = diff_data['net_diff_magnitude'][diff_data['valid_regions_mask']]
                all_diff_magnitudes.extend(valid_diffs)
        if all_diff_magnitudes:
            print(f"• temporal_evolution.png - Individual temporal analysis")
            print(f"• Complete data files for further analysis")
            print(f"\nResults saved to: {output_dir}/")
    return analyzer

if __name__ == "__main__":
    video_path = "path/to/your/mountain_slope_video.mp4"
    model_path = "rf_model.joblib"
    analyzer = analyze_video_temporal_displacement(
        video_path=video_path,
        model_path=model_path,
        interval_seconds=10,
        region_size=50,
        subset_size=16,
        step=4,
        output_dir="mountain_slope_temporal_analysis"
    )
    print("\nAnalysis completed")