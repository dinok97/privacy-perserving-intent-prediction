# data/

Default output folder for `../video_processing/video_processor.py`. The
recorder creates one dated folder per day and **appends** same-day sessions:

```
data/
  2026-09-19/
    enter/    <clip>.json  <clip>.mp4  <clip>_pose.json
    pass/
    exit/
    removed/  too_short/  too_few_points/  partial_bottom/  corrupt/
    _sessions/   session_<date>_<time>.json   (per-session metadata + counts)
    _temp/       scratch, emptied as clips finalize
```

File names use the source video stem, with `_2`, `_3`, and later suffixes for
multiple tracks from one source. Nothing is deleted —
filtered clips are sorted into `removed/<reason>/`. See
`../video_processing/README.md` for the JSON schema.

(This folder is intentionally kept in version control via this README; the
recordings themselves are data, not code.)
