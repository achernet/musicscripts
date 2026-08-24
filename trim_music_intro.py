#!/usr/bin/env python3
"""Detect a post-splash beat drop and create a safely trimmed audio copy."""

import argparse
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, stft


@dataclass(frozen=True)
class DetectionConfig:
    """Parameters for conservative intro-boundary and beat-train detection."""

    analysis_duration: float = 30.0
    minimum_intro: float = 1.0
    pre_roll: float = 0.05
    bpm_range: Tuple[float, float] = (80.0, 190.0)
    required_intervals: int = 6
    timing_tolerance: float = 0.07
    quiet_boundary: Tuple[float, float] = (0.30, 0.45)


@dataclass(frozen=True)
class DetectionResult:
    """Detected beat time, recommended trim point, and supporting diagnostics."""

    beat_time: float
    trim_time: float
    bpm: Optional[float]
    confidence: float
    reason: str


NO_TRIM = DetectionResult(0.0, 0.0, None, 0.0, "no removable intro detected")


def _normalize_feature(values: np.ndarray) -> np.ndarray:
    """Robustly scale a nonnegative feature while limiting extreme outliers."""
    low = float(np.percentile(values, 20))
    spread = float(np.percentile(values, 95) - low)
    if spread <= 0:
        return np.zeros_like(values)
    return np.clip((values - low) / spread, 0.0, 3.0)


def _onset_curve(samples: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate a spectral-flux curve with extra weight on kick frequencies."""
    frame_size = 2048
    hop_size = 256
    frequencies, times, spectrum = stft(
        samples,
        fs=sample_rate,
        window="hann",
        nperseg=frame_size,
        noverlap=frame_size - hop_size,
        boundary=None,
        padded=False,
    )
    power = np.abs(spectrum) ** 2
    log_power = np.log1p(power * 1e5)
    spectral_flux = np.maximum(
        np.diff(log_power, axis=1, prepend=log_power[:, :1]),
        0.0,
    ).mean(axis=0)
    bass_power = power[(frequencies >= 35) & (frequencies <= 180)].mean(axis=0)
    bass_flux = np.maximum(
        np.diff(np.log1p(bass_power * 1e7), prepend=0.0),
        0.0,
    )
    onset = _normalize_feature(spectral_flux) + 1.5 * _normalize_feature(bass_flux)
    return times, onset


def _has_quiet_boundary(
    samples: np.ndarray,
    sample_rate: int,
    candidate_time: float,
    config: DetectionConfig,
) -> Tuple[bool, float]:
    """Check for an energy boundary immediately before a candidate beat drop."""
    quiet_window, quiet_ratio = config.quiet_boundary
    pre_start = max(0, int((candidate_time - quiet_window) * sample_rate))
    pre_end = max(pre_start + 1, int((candidate_time - 0.05) * sample_rate))
    post_start = int(candidate_time * sample_rate)
    post_end = min(len(samples), int((candidate_time + 0.75) * sample_rate))
    if post_end <= post_start:
        return False, 0.0

    pre_rms = math.sqrt(float(np.mean(samples[pre_start:pre_end] ** 2)) + 1e-12)
    post_rms = math.sqrt(float(np.mean(samples[post_start:post_end] ** 2)) + 1e-12)
    ratio = pre_rms / max(post_rms, 1e-12)
    confidence = float(np.clip(1.0 - ratio, 0.0, 1.0))
    return post_rms > 1e-4 and ratio <= quiet_ratio, confidence


def _targets_present(
    peak_times: np.ndarray,
    targets: np.ndarray,
    tolerance: float,
) -> bool:
    """Return whether every target time has a nearby detected onset."""
    return all(
        np.any(np.abs(peak_times - target) <= tolerance)
        for target in targets
    )


def _matching_period(
    candidate_index: int,
    peak_times: np.ndarray,
    config: DetectionConfig,
) -> Optional[float]:
    """Return a beat period when enough subsequent onsets align to a grid."""
    candidate = float(peak_times[candidate_index])
    minimum_bpm, maximum_bpm = config.bpm_range
    minimum_period = 60.0 / maximum_bpm
    maximum_period = 60.0 / minimum_bpm
    later = peak_times[candidate_index + 1 :]
    possible_periods = later[
        (later - candidate >= minimum_period)
        & (later - candidate <= maximum_period)
    ] - candidate

    for period in possible_periods:
        earlier = peak_times[:candidate_index]
        preceding_targets = candidate - np.arange(1, 3) * float(period)
        if _targets_present(earlier, preceding_targets, config.timing_tolerance):
            continue

        following_targets = candidate + (
            np.arange(1, config.required_intervals + 1) * float(period)
        )
        if _targets_present(later, following_targets, config.timing_tolerance):
            return float(period)
    return None


def _find_onset_peaks(
    mono: np.ndarray,
    sample_rate: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return onset features, their peak frames, and a strong-onset threshold."""
    times, onset = _onset_curve(mono, sample_rate)
    hop_duration = float(times[1] - times[0])
    minimum_distance = max(1, int(0.18 / hop_duration))
    peaks, _ = find_peaks(onset, distance=minimum_distance, prominence=0.35)
    strong_threshold = max(2.0, float(np.percentile(onset, 95)))
    return times, onset, peaks, strong_threshold


def detect_music_start_samples(
    samples: np.ndarray,
    sample_rate: int,
    config: DetectionConfig = DetectionConfig(),
) -> DetectionResult:
    """Detect the first stable beat train following a quiet intro boundary."""
    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if len(mono) < 2048:
        return NO_TRIM

    times, onset, peaks, strong_threshold = _find_onset_peaks(mono, sample_rate)
    peak_times = times[peaks]

    for peak_position, frame_index in enumerate(peaks):
        candidate = float(times[frame_index])
        if candidate < config.minimum_intro or onset[frame_index] < strong_threshold:
            continue
        period = _matching_period(peak_position, peak_times, config)
        if period is None:
            continue
        has_boundary, boundary_confidence = _has_quiet_boundary(
            mono,
            sample_rate,
            candidate,
            config,
        )
        if has_boundary:
            return DetectionResult(
                beat_time=candidate,
                trim_time=max(0.0, candidate - config.pre_roll),
                bpm=60.0 / period,
                confidence=boundary_confidence,
                reason="stable beat train after a quiet intro boundary",
            )
    return NO_TRIM


def detect_music_start(
    input_path: Path,
    config: DetectionConfig = DetectionConfig(),
) -> DetectionResult:
    """Read the beginning of an audio file and detect a safe trim point."""
    with sf.SoundFile(input_path) as audio_file:
        frame_count = int(config.analysis_duration * audio_file.samplerate)
        samples = audio_file.read(frame_count, dtype="float32", always_2d=True)
        return detect_music_start_samples(samples, audio_file.samplerate, config)


def default_output_path(input_path: Path) -> Path:
    """Return a non-destructive output name beside the source file."""
    return input_path.with_name(f"{input_path.stem}.trimmed{input_path.suffix}")


def _encoder_arguments(output_path: Path, stream_copy: bool) -> List[str]:
    """Select a high-quality encoder suitable for the requested container."""
    if stream_copy:
        return ["-c:a", "copy"]
    encoders = {
        ".flac": ["-c:a", "flac"],
        ".m4a": ["-c:a", "aac", "-q:a", "2"],
        ".mp3": ["-c:a", "libmp3lame", "-q:a", "0"],
        ".ogg": ["-c:a", "libvorbis", "-q:a", "8"],
        ".wav": ["-c:a", "pcm_s24le"],
    }
    try:
        return encoders[output_path.suffix.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported output format: {output_path.suffix}") from error


def trim_audio(
    input_path: Path,
    output_path: Path,
    trim_time: float,
    *,
    overwrite: bool = False,
    stream_copy: bool = False,
) -> None:
    """Create a trimmed copy with metadata and embedded artwork preserved."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required but was not found")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("output must differ from input; the source is never overwritten")

    command: List[str] = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-ss",
        f"{trim_time:.6f}",
        "-map",
        "0:a:0",
        "-map",
        "0:v?",
        "-map_metadata",
        "0",
        *_encoder_arguments(output_path, stream_copy),
        "-c:v",
        "copy",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Audio file to analyze")
    parser.add_argument("-o", "--output", type=Path, help="Trimmed output path")
    parser.add_argument("--detect-only", action="store_true", help="Report without trimming")
    parser.add_argument("--pre-roll", type=float, default=0.05, help="Seconds before first beat")
    parser.add_argument("--analysis-duration", type=float, default=30.0)
    parser.add_argument("--force", action="store_true", help="Replace an existing output")
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="Use sample-accurate re-encoding instead of lossless frame-level trimming",
    )
    return parser.parse_args()


def main() -> int:
    """Detect an intro and optionally create a trimmed copy."""
    args = parse_args()
    config = DetectionConfig(
        analysis_duration=args.analysis_duration,
        pre_roll=max(0.0, args.pre_roll),
    )
    try:
        result = detect_music_start(args.input, config)
    except (OSError, RuntimeError) as error:
        print(f"Could not analyze {args.input}: {error}")
        return 1

    print(
        f"beat={result.beat_time:.3f}s trim={result.trim_time:.3f}s "
        f"bpm={result.bpm if result.bpm else 'unknown'} "
        f"confidence={result.confidence:.2f} ({result.reason})"
    )
    if args.detect_only or result.trim_time <= 0:
        return 0

    output_path = args.output or default_output_path(args.input)
    stream_copy = (
        not args.reencode
        and output_path.suffix.lower() == args.input.suffix.lower()
    )
    try:
        trim_audio(
            args.input,
            output_path,
            result.trim_time,
            overwrite=args.force,
            stream_copy=stream_copy,
        )
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Could not create trimmed file: {error}")
        return 1
    print(f"Created {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
