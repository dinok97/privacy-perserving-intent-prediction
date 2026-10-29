# data/

Default output folder for `../processing/video_processor.py`. The
recorder creates one dated folder per day and **appends** same-day sessions:

```
data/
  2026-09-19/
    <clip>.mp4  <clip>.json  <clip>_pose.json   one set per clip
    <clip>.part.mp4                             clip still being written (or left by a crash)
    clips.csv                                   one row per clip: label, cleaning result, files
    session_<date>_<time>.json                  per-session metadata + counts
```

File names use the source video stem, with `_2`, `_3`, and later suffixes for
multiple tracks from one source. Nothing is deleted —
filtered clips are sorted into `removed/<reason>/`. See
`../processing/README.md` for the JSON schema.

(This folder is intentionally kept in version control via this README; the
recordings themselves are data, not code.)
