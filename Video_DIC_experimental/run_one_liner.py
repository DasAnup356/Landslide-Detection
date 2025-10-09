# One-liner: run video analysis
from video_analysis import analyze_video_temporal_displacement
 
video_path = "./Test_files/Video1.mp4"
model_path = "rf_model.joblib"

analyzer = analyze_video_temporal_displacement(
    video_path=video_path,
    model_path=model_path,
    interval_seconds=10,
    region_size=50
)
