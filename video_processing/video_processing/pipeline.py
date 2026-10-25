"""Pipeline entry points for the video-processing workflow.

This module coordinates the sequence of preparation and analysis steps used to
process source videos, track motion, segment backgrounds, and extract features.
The execution order is intentionally kept simple and follows a linear pipeline:
1. correct fisheye distortion,
2. run the trajectory tracker,
3. create and segment background images,
4. extract features from tracked outputs.
"""

from pathlib import Path
import json

import fisheye_correction 
import video_processor
import background_extraction
import background_segmentation
import feature_extractor


SOURCE_DIR = Path(__file__).resolve().parent.parent / "source"


def run_fisheye_corrector():
    """Correct fisheye distortion for all supported source videos.

    The function scans the source directory (../source) for video files and passes the
    resulting list to the fisheye correction module for processing.
    """
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    ORIGINAL_VIDEOS = sorted(
        str(path)
        for path in SOURCE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ) if SOURCE_DIR.is_dir() else []

    fisheye_correction.correct_videos(ORIGINAL_VIDEOS)


def run_trajectory_tracker():
    """
    Read clips from `../source` or `../source/[TEMP]_undistorted_results`.
    Then put the results into `../data`.
    This script also produce `../source/processed_video_paths.json`
    """
    video_processor.run()

    
def run_background_extractor_segmentation():
    background_paths = []
    processed_video_path = Path(f"{SOURCE_DIR}/processed_video_paths.json")
    with open(processed_video_path) as f:
        video_paths = json.load(f)
        
    for p in video_paths:
        background_path = background_extraction.median_background_over_a_video(p)
        background_paths.append(background_path)
    
    background_segmentation.segment_images(background_paths)


def run_feature_extractor():
    with open(f"{SOURCE_DIR}/processed_video_paths.json") as f:
        processed_video_paths = json.load(f)
    
    trajectory_paths = []
    for p in processed_video_paths:
        traj_path = Path(p).with_name(f"{Path(p).stem}.json")
        trajectory_paths.append(traj_path)
    
    for tp in trajectory_paths:
        feature_extractor.extract_features(tp, debug=True)


def run_pipeline():
    run_fisheye_corrector()
    run_trajectory_tracker()
    run_background_extractor_segmentation()
    run_feature_extractor()
    

if __name__ == "__main__":
    run_pipeline()