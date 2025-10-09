import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import argparse
import json
from datetime import datetime

class VegetationBoundaryTool:
    def __init__(self, image_path, output_dir):
        self.image_path = image_path
        self.output_dir = output_dir
        self.image_name = Path(image_path).stem
        
        # Load original image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.height, self.width = self.original_image.shape[:2]
        print(f"Loaded image: {self.width}x{self.height}")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.patches_dir = os.path.join(output_dir, f"{self.image_name}_patches")
        os.makedirs(self.patches_dir, exist_ok=True)
        
        # Initialize drawing variables
        self.drawing = False
        self.polygons = []  # List of polygons (vegetation areas)
        self.current_polygon = []
        self.display_image = None
        self.scale_factor = 1.0
        
        # Create display image (scaled if needed)
        self.setup_display()
        
        print("\n🎨 DRAWING INSTRUCTIONS:")
        print("  • Left click to add points to polygon")
        print("  • Right click to finish current polygon")
        print("  • Press 'z' to undo last point")
        print("  • Press 'c' to clear all polygons")
        print("  • Press 'u' to undo last polygon")
        print("  • Press 's' to save and process patches")
        print("  • Press 'q' to quit without saving")
        print("  • Press SPACE to toggle polygon fill view")

    def setup_display(self):
        """Setup display image with appropriate scaling"""
        max_display_size = 1000  # Maximum display dimension
        
        if max(self.width, self.height) > max_display_size:
            self.scale_factor = max_display_size / max(self.width, self.height)
            new_width = int(self.width * self.scale_factor)
            new_height = int(self.height * self.scale_factor)
            self.display_image = cv2.resize(self.original_image, (new_width, new_height))
            print(f"Scaled display to: {new_width}x{new_height} (scale: {self.scale_factor:.3f})")
        else:
            self.display_image = self.original_image.copy()
            
        self.current_display = self.display_image.copy()

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for polygon drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point to current polygon
            # Convert display coordinates back to original image coordinates
            orig_x = int(x / self.scale_factor)
            orig_y = int(y / self.scale_factor)
            
            self.current_polygon.append((orig_x, orig_y))
            self.update_display()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Finish current polygon
            if len(self.current_polygon) >= 3:
                self.polygons.append(self.current_polygon.copy())
                print(f"Polygon {len(self.polygons)} added with {len(self.current_polygon)} points")
                self.current_polygon = []
                self.update_display()

    def update_display(self):
        """Update the display with current polygons"""
        self.current_display = self.display_image.copy()
        
        # Draw completed polygons
        for i, polygon in enumerate(self.polygons):
            # Scale polygon points for display
            display_polygon = [(int(p[0] * self.scale_factor), int(p[1] * self.scale_factor)) 
                             for p in polygon]
            
            # Draw filled polygon
            pts = np.array(display_polygon, dtype=np.int32)
            cv2.fillPoly(self.current_display, [pts], (0, 255, 0, 128))  # Green fill
            
            # Draw polygon outline
            cv2.polylines(self.current_display, [pts], True, (0, 255, 0), 2)
            
            # Add polygon number
            if display_polygon:
                cv2.putText(self.current_display, f"V{i+1}", display_polygon[0], 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw current polygon being drawn
        if len(self.current_polygon) > 0:
            display_current = [(int(p[0] * self.scale_factor), int(p[1] * self.scale_factor)) 
                             for p in self.current_polygon]
            
            # Draw points
            for point in display_current:
                cv2.circle(self.current_display, point, 4, (0, 0, 255), -1)
            
            # Draw lines between points
            if len(display_current) > 1:
                pts = np.array(display_current, dtype=np.int32)
                cv2.polylines(self.current_display, [pts], False, (0, 0, 255), 2)

    def create_vegetation_mask(self):
        """Create binary mask for vegetation areas"""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        for polygon in self.polygons:
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        
        return mask

    def calculate_optimal_patches(self, target_patches=3500, patch_size=(30, 35)):
        """Calculate optimal patching strategy"""
        patch_h, patch_w = patch_size
        
        # Method 1: Fixed patch size
        patches_h_fixed = self.height // patch_h
        patches_w_fixed = self.width // patch_w
        total_patches_fixed = patches_h_fixed * patches_w_fixed
        
        # Method 2: Target number of patches
        aspect_ratio = self.width / self.height
        target_aspect = patch_w / patch_h
        
        patch_h_target = int(np.sqrt(self.height * self.width / (target_patches * target_aspect)))
        patch_w_target = int(patch_h_target * target_aspect)
        
        patch_h_target = min(patch_h_target, self.height)
        patch_w_target = min(patch_w_target, self.width)
        
        patches_h_target = self.height // patch_h_target
        patches_w_target = self.width // patch_w_target
        total_patches_target = patches_h_target * patches_w_target
        
        # Choose method with minimum patches
        if total_patches_fixed <= total_patches_target:
            return {
                'method': 'fixed_size',
                'patch_size': (patch_h, patch_w),
                'grid_size': (patches_h_fixed, patches_w_fixed),
                'total_patches': total_patches_fixed
            }
        else:
            return {
                'method': 'target_count', 
                'patch_size': (patch_h_target, patch_w_target),
                'grid_size': (patches_h_target, patches_w_target),
                'total_patches': total_patches_target
            }

    def extract_and_label_patches(self):
        """Extract patches and automatically label them based on vegetation mask"""
        if not self.polygons:
            print("❌ No vegetation areas marked! Please draw at least one polygon.")
            return
        
        print(f"\n🔄 Processing patches for {self.image_name}...")
        
        # Create vegetation mask
        vegetation_mask = self.create_vegetation_mask()
        
        # Calculate optimal patch strategy
        patch_info = self.calculate_optimal_patches()
        patch_h, patch_w = patch_info['patch_size']
        patches_h, patches_w = patch_info['grid_size']
        
        print(f"Using {patch_info['method']}: {patch_info['total_patches']} patches of size {patch_h}x{patch_w}")
        
        # Extract patches and labels
        patch_data = []
        vegetation_count = 0
        non_vegetation_count = 0
        
        for i in range(patches_h):
            for j in range(patches_w):
                # Calculate patch coordinates
                y1 = i * patch_h
                y2 = y1 + patch_h
                x1 = j * patch_w
                x2 = x1 + patch_w
                
                # Extract patch from original image
                patch = self.original_image[y1:y2, x1:x2]
                
                # Extract corresponding mask patch
                mask_patch = vegetation_mask[y1:y2, x1:x2]
                
                # Determine label based on vegetation coverage
                vegetation_pixels = np.sum(mask_patch > 0)
                total_pixels = mask_patch.size
                vegetation_ratio = vegetation_pixels / total_pixels
                
                # Label as vegetation if >50% of patch is vegetation
                label = 1 if vegetation_ratio > 0.5 else 0
                
                if label == 1:
                    vegetation_count += 1
                else:
                    non_vegetation_count += 1
                
                # Save patch image
                patch_filename = f"{self.image_name}_patch_{i:03d}_{j:03d}.jpg"
                patch_path = os.path.join(self.patches_dir, patch_filename)
                cv2.imwrite(patch_path, patch)
                
                # Store patch metadata
                patch_data.append({
                    'filename': patch_filename,
                    'original_image': self.image_name,
                    'row': i,
                    'col': j,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'label': label,
                    'vegetation_ratio': round(vegetation_ratio, 3),
                    'patch_size_h': patch_h,
                    'patch_size_w': patch_w
                })
        
        # Save to CSV
        df = pd.DataFrame(patch_data)
        csv_path = os.path.join(self.output_dir, f"{self.image_name}_labels.csv")
        df.to_csv(csv_path, index=False)
        
        # Save polygon data
        polygon_data = {
            'image_name': self.image_name,
            'image_path': self.image_path,
            'image_size': [self.width, self.height],
            'polygons': self.polygons,
            'patch_info': patch_info,
            'timestamp': datetime.now().isoformat()
        }
        
        polygon_path = os.path.join(self.output_dir, f"{self.image_name}_polygons.json")
        with open(polygon_path, 'w') as f:
            json.dump(polygon_data, f, indent=2)
        
        # Save visualization
        vis_image = self.original_image.copy()
        for polygon in self.polygons:
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(vis_image, [pts], (0, 255, 0), cv2.LINE_AA)
            cv2.polylines(vis_image, [pts], True, (0, 255, 0), 3)
        
        vis_path = os.path.join(self.output_dir, f"{self.image_name}_vegetation_mask.jpg")
        cv2.imwrite(vis_path, vis_image)
        
        print(f"\n✅ Processing complete!")
        print(f"   📁 Patches saved to: {self.patches_dir}")
        print(f"   📊 Labels saved to: {csv_path}")
        print(f"   🗺️ Polygons saved to: {polygon_path}")
        print(f"   🖼️ Visualization saved to: {vis_path}")
        print(f"\n📈 Statistics:")
        print(f"   Total patches: {len(patch_data)}")
        print(f"   Vegetation patches: {vegetation_count}")
        print(f"   Non-vegetation patches: {non_vegetation_count}")
        print(f"   Vegetation ratio: {vegetation_count/len(patch_data):.1%}")

    def run(self):
        """Run the interactive boundary marking tool"""
        window_name = f"Vegetation Boundary Tool - {self.image_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.update_display()
        
        while True:
            cv2.imshow(window_name, self.current_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting without saving...")
                break
            elif key == ord('z'):
                # Undo last point in current polygon
                if self.current_polygon:
                    removed_point = self.current_polygon.pop()
                    print(f"Removed last point: {removed_point}")
                    self.update_display()
                else:
                    print("No points to remove in current polygon")
            elif key == ord('c'):
                # Clear all polygons
                self.polygons = []
                self.current_polygon = []
                self.update_display()
                print("All polygons cleared")
            elif key == ord('u'):
                # Undo last polygon
                if self.polygons:
                    removed = self.polygons.pop()
                    print(f"Removed polygon with {len(removed)} points")
                    self.update_display()
                else:
                    print("No polygons to remove")
            elif key == ord('s'):
                # Save and process
                self.extract_and_label_patches()
                break
            elif key == ord(' '):
                # Toggle polygon visibility (just for viewing)
                if len(self.polygons) > 0:
                    temp_display = self.display_image.copy()
                    cv2.imshow(window_name, temp_display)
                    cv2.waitKey(500)  # Show for 500ms
                    self.update_display()
        
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Vegetation boundary marking and auto-labeling tool')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('output_dir', help='Directory to save patches and labels')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file {args.image_path} does not exist")
        return
    
    try:
        tool = VegetationBoundaryTool(args.image_path, args.output_dir)
        tool.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

# Example usage:
# python vegetation_boundary_tool.py /path/to/image.jpg /path/to/output/directory