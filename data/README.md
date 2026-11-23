# data/

Default output folder for `../processing/video_processor.py`. The
tracker writes one flat folder, so there is exactly one `clips.csv` per output
folder:

```
data/
  <clip>.mp4  <clip>.json  <clip>_pose.json   one set per clip
  <clip>.part.mp4                             clip still being written (or left by a crash)
  clips.csv                                   one row per clip: label, cleaning result, files
  session_<date>_<time>.json                  per-session metadata + counts
```

A new run on the same folder **replaces** the `clips.csv` row and the files of every 
clip it produces (rows are matched by `clip_id`). Rows of clips the new run does not 
produce stay as they were. There is no per-day subfolder any more, so a failed or 
repeated run can no longer leave a second `clips.csv` with the same `clip_id`s (the 
validator reports that as `TRACKS_DUPLICATE_ROW`). To start from nothing, use an empty 
output folder.

File names use the source video stem, with `_2`, `_3`, and later suffixes for
multiple tracks from one source. Nothing is deleted —
filtered clips are sorted into `removed/<reason>/`. See
`../processing/README.md` for the JSON schema.

(This folder is intentionally kept in version control via this README; the
recordings themselves are data, not code.)