"""Recorded local paths remain usable after flattening the project layout."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import project_paths
from geometry3d import pipeline_geometry


class RelocatedPaths(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.file = self.root / 'data/run/processing_manifest.json'
        self.file.parent.mkdir(parents=True)
        self.file.write_text('{}')
        self.patch = patch.object(project_paths, 'ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_legacy_absolute_metadata_maps_to_the_moved_file(self):
        old = self.root / 'video_processing/data/run/processing_manifest.json'
        self.assertEqual(project_paths.resolve_project_path(old), self.file)

    def test_old_repo_relative_metadata_works_with_a_manifest_base(self):
        self.assertEqual(project_paths.resolve_project_path(
            'video_processing/data/run/processing_manifest.json', base=self.root / 'data/inputs'), self.file)

    def test_new_manifest_relative_paths_keep_their_base(self):
        self.assertEqual(project_paths.resolve_project_path(
            'run/processing_manifest.json', base=self.root / 'data'), self.file)

    def test_external_checkout_paths_are_not_rewritten(self):
        external = self.root.parent / 'another-project/video_processing/checkpoint.pt'
        self.assertEqual(project_paths.resolve_project_path(external), external.resolve())

    def test_manifest_argument_is_relative_to_the_callers_working_directory(self):
        video = self.file.with_name('clip.mp4')
        trajectory = self.file.with_name('track.json')
        video.touch()
        trajectory.write_text('{}')
        manifest = self.root / 'data/inputs.json'
        manifest.write_text(json.dumps([{'video': 'run/clip.mp4', 'trajectory': 'run/track.json'}]))
        previous = Path.cwd()
        try:
            os.chdir(manifest.parent)
            record = pipeline_geometry.records_from_manifest('inputs.json')[0]
        finally:
            os.chdir(previous)
        self.assertEqual(record['video'], video)
        self.assertEqual(record['trajectory'], trajectory)


if __name__ == '__main__':
    unittest.main()
