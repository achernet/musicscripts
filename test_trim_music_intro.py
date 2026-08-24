"""Tests for conservative splash-intro detection and trimming helpers."""

import unittest
from pathlib import Path

import numpy as np

from trim_music_intro import (
    DetectionConfig,
    default_output_path,
    detect_music_start_samples,
)


def _kick_train(
    sample_rate: int,
    duration: float,
    first_beat: float,
    period: float = 0.5,
) -> np.ndarray:
    """Create a synthetic low-frequency beat train with decaying transients."""
    samples = np.zeros(int(sample_rate * duration), dtype=np.float32)
    beat_length = int(0.12 * sample_rate)
    beat_time = np.arange(beat_length) / sample_rate
    kick = np.sin(2 * np.pi * 65 * beat_time) * np.exp(-beat_time * 30)
    for onset in np.arange(first_beat, duration - 0.1, period):
        start = int(onset * sample_rate)
        samples[start : start + beat_length] += kick.astype(np.float32)
    return samples


class MusicStartDetectionTests(unittest.TestCase):
    """Verify beat-drop detection remains conservative and precise."""

    def test_detects_beat_train_after_splash_and_quiet_boundary(self):
        """A splash followed by quiet and regular kicks should be trimmed."""
        sample_rate = 22050
        samples = _kick_train(sample_rate, 8.0, first_beat=2.5)
        splash_start = int(0.1 * sample_rate)
        splash_end = int(2.0 * sample_rate)
        splash_time = np.arange(splash_end - splash_start) / sample_rate
        samples[splash_start:splash_end] += (
            0.12 * np.sin(2 * np.pi * 330 * splash_time)
        ).astype(np.float32)

        result = detect_music_start_samples(samples, sample_rate)

        self.assertAlmostEqual(result.beat_time, 2.5, delta=0.08)
        self.assertAlmostEqual(result.trim_time, 2.45, delta=0.08)
        self.assertAlmostEqual(result.bpm or 0, 120.0, delta=3.0)

    def test_does_not_trim_music_that_starts_immediately(self):
        """A normal beat train beginning at the start must remain untouched."""
        sample_rate = 22050
        samples = _kick_train(sample_rate, 8.0, first_beat=0.1)

        result = detect_music_start_samples(samples, sample_rate)

        self.assertEqual(result.trim_time, 0.0)

    def test_short_or_nonmusical_audio_is_not_trimmed(self):
        """Insufficient or nonperiodic material must not produce a trim point."""
        sample_rate = 22050
        samples = np.zeros(sample_rate * 4, dtype=np.float32)
        samples[sample_rate : sample_rate + 100] = 0.5

        result = detect_music_start_samples(
            samples,
            sample_rate,
            DetectionConfig(required_intervals=4),
        )

        self.assertEqual(result.trim_time, 0.0)

    def test_default_output_does_not_replace_source(self):
        """The default path should clearly identify a new trimmed copy."""
        source = Path("Artist - Song.mp3")
        self.assertEqual(
            default_output_path(source),
            Path("Artist - Song.trimmed.mp3"),
        )


if __name__ == "__main__":
    unittest.main()
