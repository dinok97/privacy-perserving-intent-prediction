"""Tests for cleanup of root-level source videos."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from processing import clean_datasource


class CleanVideos(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'source'
        self.source.mkdir()

    def video(self, name, content):
        path = self.source / name
        path.write_bytes(content)
        return path

    def test_moves_duplicates_and_short_files_but_keeps_nested_videos(self):
        kept = self.video('a.mp4', b'identical')
        duplicate = self.video('b.mp4', b'identical')
        short = self.video('short.MKV', b'short')
        boundary = self.video('boundary.mov', b'boundary')
        nested_dir = self.source / 'nested'
        nested_dir.mkdir()
        nested = nested_dir / 'nested.mp4'
        nested.write_bytes(b'nested')

        def capture_for(path):
            capture = Mock()
            capture.isOpened.return_value = True
            durations = {'a.mp4': (30, 30), 'short.MKV': (14, 30),
                         'boundary.mov': (15, 30)}
            frames, fps = durations[Path(path).name]
            capture.get.side_effect = lambda prop: (
                frames if prop == clean_datasource.cv2.CAP_PROP_FRAME_COUNT else fps)
            return capture

        with patch.object(clean_datasource.cv2, 'VideoCapture', side_effect=capture_for):
            result = clean_datasource.clean_videos(self.source)

        self.assertEqual(result['duplicates'], [self.source / 'deleted/duplicated/b.mp4'])
        self.assertEqual(result['too_short'], [self.source / 'deleted/too_short/short.MKV'])
        self.assertEqual(result['low_fps'], [])
        self.assertTrue(kept.exists())
        self.assertTrue(boundary.exists())
        self.assertTrue(nested.exists())
        self.assertFalse(duplicate.exists())
        self.assertFalse(short.exists())

    def test_moves_videos_below_five_fps_but_keeps_boundary(self):
        low_fps = self.video('low.mp4', b'low fps')
        boundary = self.video('boundary.mp4', b'five fps')

        def capture_for(path):
            capture = Mock()
            capture.isOpened.return_value = True
            fps = 4.9 if Path(path).name == 'low.mp4' else 5.0
            capture.get.side_effect = lambda prop: (
                fps * 2 if prop == clean_datasource.cv2.CAP_PROP_FRAME_COUNT else fps)
            return capture

        with patch.object(clean_datasource.cv2, 'VideoCapture', side_effect=capture_for):
            result = clean_datasource.clean_videos(self.source)

        self.assertEqual(result['low_fps'], [self.source / 'deleted/low_fps/low.mp4'])
        self.assertTrue(boundary.exists())
        self.assertFalse(low_fps.exists())

    def test_unreadable_video_raises_without_moving_it(self):
        video = self.video('broken.mp4', b'not a video')
        capture = Mock()
        capture.isOpened.return_value = False

        with patch.object(clean_datasource.cv2, 'VideoCapture', return_value=capture):
            with self.assertRaisesRegex(RuntimeError, 'Cannot open video'):
                clean_datasource.clean_videos(self.source)

        self.assertTrue(video.exists())
        capture.release.assert_called_once()

    def test_existing_destination_is_not_overwritten(self):
        self.video('a.mp4', b'identical')
        duplicate = self.video('b.mp4', b'identical')
        destination = self.source / 'deleted/duplicated/b.mp4'
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b'preserve this')
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: (
            30 if prop == clean_datasource.cv2.CAP_PROP_FRAME_COUNT else 30)

        with patch.object(clean_datasource.cv2, 'VideoCapture', return_value=capture) as open_video:
            with self.assertRaises(FileExistsError):
                clean_datasource.clean_videos(self.source)

        self.assertEqual(destination.read_bytes(), b'preserve this')
        self.assertTrue(duplicate.exists())
        open_video.assert_called_once()


if __name__ == '__main__':
    unittest.main()
