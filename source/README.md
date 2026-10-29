# Video source

Place prerecorded `.mp4`, `.avi`, `.mov`, `.mkv`, or `.m4v` files in this directory.
The collector processes the first supported video in sorted filename order.

Video files are ignored by git because they may contain identifiable recordings.

The `simple_person_recorder.py` can be used to capture your own source videos, which are saved in this folder. It automatically detects a person using the `yolo11` model.
