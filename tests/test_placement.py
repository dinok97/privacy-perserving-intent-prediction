import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from processing import feature_extractor, fisheye_correction, placement


def fake_segment(background):
    Path(background).with_name(placement.SCENE).write_text('{"image_size": [64, 48], "objects": []}')


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.directory = self.root / 'placement'
        self.videos = [self.root / 'a.mp4', self.root / 'b.mp4', self.root / 'c.mp4']
        self.image = np.full((48, 64, 3), 90, np.uint8)
        self.report = {'people_left': 0}
        self.patches = [
            patch.object(placement, 'interactive', return_value=True),
            patch.object(placement, 'video_size', return_value=[64, 48]),
            patch.object(placement, 'candidate', side_effect=lambda v, u: (self.image, None, self.report)),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def prepare(self, answers, undistort=False, videos=None):
        with patch.object(placement, 'ask', side_effect=answers) as ask, patch('builtins.print'):
            result = placement.prepare(videos or self.videos, self.directory, undistort, fake_segment)
        return result, ask

    def test_needs_a_terminal(self):
        with patch.object(placement, 'interactive', return_value=False):
            with self.assertRaisesRegex(placement.PlacementError, 'terminal'):
                placement.prepare(self.videos, self.directory, False, fake_segment)

    def test_rejected_background_moves_on_to_the_next_video(self):
        result, ask = self.prepare([False, True])
        self.assertEqual(ask.call_count, 2)
        self.assertFalse((self.directory / placement.CANDIDATE).exists())
        self.assertTrue(result['background'].is_file() and result['scene'].is_file())
        meta = json.loads((self.directory / placement.META).read_text())
        self.assertEqual(meta['source_video'], 'b.mp4')
        self.assertEqual(meta['image_size'], [64, 48])

    def test_all_rejected_stops_the_run(self):
        with self.assertRaisesRegex(placement.PlacementError, 'approved'):
            self.prepare([False, False, False])
        self.assertFalse((self.directory / placement.META).exists())
        self.assertFalse((self.directory / placement.CANDIDATE).exists())

    def test_existing_background_still_asks_and_is_reused_on_yes(self):
        self.prepare([True])
        with patch.object(placement, 'candidate', side_effect=AssertionError('must not rebuild')):
            result, ask = self.prepare([True])
        self.assertEqual(ask.call_count, 1)
        self.assertTrue(result['reused'])

    def test_existing_background_rejected_makes_a_new_one(self):
        self.prepare([True])
        result, ask = self.prepare([False, True])
        self.assertEqual(ask.call_count, 2)
        self.assertFalse(result['reused'])

    def test_other_image_size_stops_the_run(self):
        self.prepare([True])
        with patch.object(placement, 'video_size', return_value=[128, 96]):
            with self.assertRaisesRegex(placement.PlacementError, 'another camera setup'):
                self.prepare([True])

    def test_existing_placement_must_match_the_undistortion_setting(self):
        self.prepare([True], undistort=True)
        with self.assertRaisesRegex(placement.PlacementError, 'undistortion'):
            self.prepare([True], undistort=False)
        self.prepare([True], undistort=None)  # already tracked videos accept it


class LensTests(unittest.TestCase):
    def test_given_lens_is_applied_without_detecting_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / 'clip.mp4'
            video.write_bytes(b'x')
            lens = {'image_size': [64, 48], 'distorted': False, 'reason': 'shared', 'n_chains': 9}
            with patch.object(fisheye_correction, 'median_background', side_effect=AssertionError('detected')):
                out = fisheye_correction.correct_video(video, lens=lens)
            saved = json.loads((Path(out).parent / 'clip_lens.json').read_text())
            self.assertEqual(saved['reason'], 'shared')
            self.assertEqual(saved['video'], 'clip.mp4')


class BrightnessTests(unittest.TestCase):
    def test_person_box_is_left_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'clip.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 5, (64, 48))
            frame = np.full((48, 64, 3), 100, np.uint8)
            frame[10:30, 20:40] = 255  # a bright "person"
            for _ in range(20):
                writer.write(frame)
            writer.release()
            frames = [{'frame_index': i, 'bbox': [20, 10, 40, 30]} for i in range(20)]
            with_box, _ = feature_extractor.clip_brightness(path, frames)
            without_box, _ = feature_extractor.clip_brightness(path, [])
            self.assertAlmostEqual(with_box, 100 / 255, delta=0.02)
            self.assertGreater(without_box, with_box + 0.03)

    def test_unreadable_video_gives_none(self):
        self.assertIsNone(feature_extractor.clip_brightness('/nonexistent.mp4', []))


if __name__ == '__main__':
    unittest.main()