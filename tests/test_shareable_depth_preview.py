from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from geometry3d import shareable_depth_preview as preview


class ShareableDepthPreview(unittest.TestCase):
    def test_depth_and_rgb_patterns_do_not_appear_in_export(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'preview'
            output.mkdir()
            keypoints = np.zeros((17, 3))
            keypoints[preview.TORSO] = [[22, 21, 1], [38, 21, 1], [38, 53, 1], [22, 53, 1]]
            detected = {'bbox': [10, 10, 50, 64], 'keypoints': keypoints.tolist()}
            without_pose = {**detected, 'keypoints': None}
            tracking = {'metadata': {'frame_width': 64, 'frame_height': 64, 'frame_count': 4},
                        'frames': [{'frame_index': i, 'detections': d} for i, d in
                                   enumerate([[detected], [detected], [without_pose], []])]}
            alignment = {'source_image_size': [64, 64], 'image_size': [64, 64],
                         'crop_offset': [0, 0], 'padding': [0, 0]}
            for i in range(4):
                pattern = np.indices((64, 64)).sum(axis=0) % 2
                depth = np.where(pattern, 99.0, 78.0)
                depth[21:54, 22:39] = float(i + 1)
                np.savez(root / f'frame_{i:06d}.npz', depth=depth[..., None],
                         depth_conf=pattern, images=np.repeat(pattern[..., None], 3, axis=2) * 255)
            destination = output / 'clip_0001.mp4'
            result = preview.render_preview(root, tracking, alignment, 12.0, destination)
            self.assertEqual((result['frames'], result['frames_with_marker'], result['frames_with_depth']),
                             (4, 3, 2))
            self.assertEqual(list(output.iterdir()), [destination])

            video = cv2.VideoCapture(str(destination))
            try:
                self.assertAlmostEqual(video.get(cv2.CAP_PROP_FPS), 12.0)
                expected = [preview.DEPTH_COLOURS[0], preview.DEPTH_COLOURS[-1], preview.NEUTRAL, preview.BACKGROUND]
                for colour in expected:
                    ok, image = video.read()
                    self.assertTrue(ok)
                    # The centre shows only the chosen colour. The rest of the canvas
                    # retains its neutral background even where the input had detail.
                    np.testing.assert_allclose(image[261, 301], colour, atol=8)
                    np.testing.assert_allclose(image[80, 80], preview.BACKGROUND, atol=8)
                self.assertFalse(video.read()[0])
            finally:
                video.release()

            room = np.broadcast_to(np.linspace(0.8, 4.0, 64), (64, 64))
            np.savez(root / 'frame_000003.npz', depth=room[..., None])
            heatmap_output = output / 'room_preview.mp4'
            with self.assertRaisesRegex(ValueError, 'contains a tracked person'):
                preview.render_preview(root, tracking, alignment, 12.0, heatmap_output, background_frame=0)
            self.assertFalse(heatmap_output.exists())
            result = preview.render_preview(root, tracking, alignment, 12.0, heatmap_output, background_frame=3)
            self.assertEqual(result['background_frame'], 3)
            video = cv2.VideoCapture(str(heatmap_output))
            try:
                first_ok, first = video.read()
                video.set(cv2.CAP_PROP_POS_FRAMES, 3)
                last_ok, last = video.read()
                self.assertTrue(first_ok and last_ok)
                for x in (160, 470):
                    np.testing.assert_allclose(first[120, x], last[120, x], atol=8)
                self.assertGreater(np.abs(first[120, 160].astype(int) - first[120, 470]).max(), 40)
                self.assertGreater(np.abs(first[260, 309].astype(int) - last[260, 309]).max(), 40)
            finally:
                video.release()


if __name__ == '__main__':
    unittest.main()
