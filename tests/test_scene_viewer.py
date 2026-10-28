"""A scene generated through the API must include its offline dependencies."""
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry3d import segment_background_3d as segmentation


class OfflineSceneViewer(unittest.TestCase):
    def test_direct_api_bundles_every_local_script_referenced_by_the_viewer(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'scene'
            out.mkdir()
            scene = {'image': 'clip.png', 'image_size': [32, 24], 'objects': [
                {'label': 'floor', 'polygon': [[0, 0], [31, 0], [31, 23], [0, 23]]}]}
            labels = np.ones((14, 14), dtype=np.uint8)
            with patch.object(segmentation, 'get_plotlyjs', return_value='/* test bundle */'):
                segmentation.write_viewer(out, 'scene', np.zeros((14, 14, 3)),
                    np.zeros((14, 14, 3), dtype=np.uint8), labels, labels,
                    labels.astype(bool), np.ones((14, 14)), scene)
            scripts = re.findall(r'<script[^>]+src="([^"]+)"', (out / 'viewer.html').read_text(encoding='utf-8'))
            self.assertTrue(scripts)
            for script in scripts:
                self.assertTrue((out / script).is_file(), script)


if __name__ == '__main__':
    unittest.main()
