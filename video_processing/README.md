# video_processing

Scripts in this directory process pedestrian videos by detecting people approaching 
or leaving the door, labels each video `enter` / `pass` / `exit` and put the result in `data/` directory.

## Structure

```
video_processing/
  requirements.txt     Python dependencies (see Setup)
  check_env.py         run first: checks numpy / opencv / torch / ultralytics
   video_processing/    video processing script + its README
      video_processor.py
    README.md
  data/                where the collection script writes the dataset (default output)
```

The collection script is **self-contained** — it imports no other project code
and writes its results to `data/` by default. 

## Setup

1. Create a normal virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 check_env.py
```

2. Place one or more video files in `source/`:

```text
source/
   pedestrian_video.mp4
```

3. Run the collector from the project directory:

```bash
python3 video_processing/video_processor.py
```

The script processes every `.mp4`, `.avi`, `.mov`, `.mkv`, or `.m4v` file in
`source/` in sorted filename order, writes labeled videos and JSON files under
`data/YYYY-MM-DD/`, and does not open a GUI window. Set `source_video` to process
only one specific file. Change `source_dir`, `source_video`, `device`, or
`record_fps` in `main()` when needed. Set `distort=True` only after replacing

the embedded calibration values with calibration for the input camera. Output
filenames use the source video stem; multiple tracks from one source receive
`_2`, `_3`, and later suffixes to avoid overwriting files.

## Privacy

This records identifiable video and pose data of people; handle it accordingly.
