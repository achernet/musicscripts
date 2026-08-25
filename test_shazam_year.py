"""Offline tests for Shazam recognition and MusicBrainz year helpers."""

import math
import socket
import tempfile
import unittest
from datetime import datetime
from unittest.mock import Mock, call, patch

from path import Path

import shazam_year


def _shazam_result(artist="The Artist", title="The Song", year="1999"):
    """Build a minimal successful Shazam response."""
    return {
        "track": {
            "key": "123",
            "subtitle": artist,
            "title": title,
            "sections": [
                {
                    "metadata": [
                        {"title": "Released", "text": year},
                    ]
                }
            ],
        }
    }


class NormalizationTests(unittest.TestCase):
    """Verify parsing and canonical filename construction."""

    def test_parse_year_handles_present_and_missing_metadata(self):
        """Release metadata should be optional and never raise on bad input."""
        self.assertEqual(shazam_year._parse_year_from_shazam_data(_shazam_result()), "1999")
        self.assertIsNone(shazam_year._parse_year_from_shazam_data({}))
        self.assertIsNone(
            shazam_year._parse_year_from_shazam_data(
                {"track": {"sections": []}}
            )
        )

    def test_strip_parens_can_remove_contents_or_only_delimiters(self):
        """Both normalization modes should handle round and square brackets."""
        text = "Song (Live) [Remastered]"

        self.assertEqual(shazam_year.strip_parens(text), "Song")
        self.assertEqual(
            shazam_year.strip_parens(text, delete_text_inside=False),
            "Song Live Remastered",
        )

    def test_artist_and_title_noise_is_removed(self):
        """Mixer tags, annotations, and trailing uncertainty marks are noise."""
        self.assertEqual(
            shazam_year.normalize_artist(" The Artist {MIXED}?? "),
            "The Artist",
        )
        self.assertEqual(
            shazam_year.normalize_title(
                "Song (Live) **Tune Of The Week** {MIXED}??"
            ),
            "Song",
        )

    def test_target_filename_is_sanitized_and_keeps_extension(self):
        """Canonical names should remove leading The, quotes, and path separators."""
        filename = shazam_year.build_target_filename(
            "The Cure",
            "'Love/Song' (Live)",
            1989,
            ext=".flac",
        )

        self.assertEqual(filename, "Cure - Love - Song (1989).flac")

    def test_artist_and_title_can_be_extracted_from_filename(self):
        """Year-only mode should accept canonical names with or without a year."""
        self.assertEqual(
            shazam_year.extract_artist_title_from_filename(
                "The Artist - The Song (1999).mp3"
            ),
            ("The Artist", "The Song"),
        )
        self.assertEqual(
            shazam_year.extract_artist_title_from_filename("Artist - Song.flac"),
            ("Artist", "Song"),
        )

    def test_malformed_filename_has_a_clear_error(self):
        """Year-only failures should identify the invalid filename."""
        with self.assertRaisesRegex(ValueError, "Cannot extract artist and title"):
            shazam_year.extract_artist_title_from_filename("unknown.mp3")


class MusicBrainzQueryTests(unittest.TestCase):
    """Verify Lucene construction, timeout handling, and release filtering."""

    def test_artist_query_splits_collaborators_and_removes_leading_the(self):
        """Common collaborator separators should create alternative artists."""
        self.assertEqual(
            shazam_year._build_artist_query("The Alpha & Beta feat. Gamma"),
            '(artist:"Alpha" OR artist:"Beta" OR artist:"Gamma")',
        )

    def test_title_query_handles_splits_and_escapes_quotes(self):
        """Slash alternatives and joined titles should produce valid Lucene."""
        self.assertEqual(
            shazam_year._build_title_query("The One and Two / Three"),
            '((recording:"One" AND recording:"Two") OR recording:"Three")',
        )
        self.assertEqual(
            shazam_year._escape_lucene_phrase('A "Quoted" \\ Song'),
            'A \\"Quoted\\" \\\\ Song',
        )

    def test_timeout_context_restores_previous_socket_default(self):
        """Temporary MusicBrainz timeouts must not leak process-wide state."""
        with (
            patch.object(socket, "getdefaulttimeout", return_value=3),
            patch.object(socket, "setdefaulttimeout") as set_timeout,
        ):
            with shazam_year.musicbrainz_timeout(7):
                pass

        self.assertEqual(set_timeout.call_args_list, [call(7), call(3)])

    def test_timeout_context_wraps_socket_timeout(self):
        """Callers should receive the domain-specific timeout exception."""
        with (
            patch.object(socket, "getdefaulttimeout", return_value=None),
            patch.object(socket, "setdefaulttimeout"),
        ):
            with self.assertRaisesRegex(
                shazam_year.MusicBrainzTimeout,
                "timed out after 4s",
            ):
                with shazam_year.musicbrainz_timeout(4):
                    raise socket.timeout("network stalled")

    def test_query_recordings_filters_matching_release_and_date(self):
        """Only matching artist/title releases should determine the year."""
        release = {
            "date": "1995-02-03",
            "medium-list": [{"track-list": [{"title": "The Song"}]}],
        }
        recording = {
            "artist-credit": [{"name": "The Artist"}],
            "release-list": [release],
        }
        responses = [
            {"recording-list": [recording]},
            {"recording-list": []},
        ]

        with patch.object(
            shazam_year.musicbrainzngs,
            "search_recordings",
            side_effect=responses,
        ) as search:
            result = shazam_year._query_recordings("The Artist", "The Song")

        self.assertEqual(result["reldate"], datetime(1995, 2, 3))
        self.assertEqual(result["recdate"], datetime(1995, 2, 3))
        self.assertEqual(result["releases"], [release])
        self.assertEqual(search.call_count, 2)


class RecognitionTests(unittest.TestCase):
    """Verify network wrappers and fallback behavior without external calls."""

    def test_medley_title_uses_earliest_slash_alternative_year(self):
        """Each title in a slash-separated Shazam medley should affect the year."""
        good_good_lovin_release = {
            "date": "1960-01-01",
            "medium-list": [
                {"track-list": [{"title": "Good Good Lovin'"}]}
            ],
        }
        sweet_little_16_release = {
            "date": "1958-01-01",
            "medium-list": [
                {"track-list": [{"title": "Sweet Little 16"}]}
            ],
        }
        recordings = [
            {
                "artist-credit": [{"name": "James Brown"}],
                "release-list": [good_good_lovin_release],
            },
            {
                "artist-credit": [{"name": "Chuck Berry"}],
                "release-list": [sweet_little_16_release],
            },
        ]
        shazam_result = _shazam_result(
            artist="James Brown & Chuck Berry",
            title="Good Good Lovin’ / Sweet Little 16",
            year="2000",
        )

        with (
            patch.object(
                shazam_year,
                "recognize_audio",
                return_value=shazam_result,
            ),
            patch.object(
                shazam_year.musicbrainzngs,
                "search_recordings",
                side_effect=[
                    {"recording-list": recordings},
                    {"recording-list": []},
                ],
            ) as search,
        ):
            result = shazam_year.query_shazam("sweet-little-sixteen.mp3")

        self.assertEqual(result["year"], 1958)
        search.assert_any_call(
            offset=0,
            limit=100,
            query=(
                '(artist:"James Brown" OR artist:"Chuck Berry") AND '
                '(recording:"Good Good Lovin" OR recording:"Sweet Little 16")'
            ),
            strict=True,
        )

    def test_songrec_failure_uses_local_fallback(self):
        """The local ShazamAPI backend should run after songrec fails."""
        expected = _shazam_result()

        with (
            patch.object(
                shazam_year,
                "fetch_shazam_data",
                side_effect=RuntimeError("songrec failed"),
            ),
            patch.object(
                shazam_year,
                "fetch_local_shazam_data",
                return_value=expected,
            ) as local,
        ):
            result = shazam_year.recognize_audio("song.mp3")

        self.assertIs(result, expected)
        local.assert_called_once_with(
            "song.mp3",
            match_count=shazam_year.DEFAULT_MATCH_COUNT,
            max_time_seconds=shazam_year.DEFAULT_SHAZAM_INTERVAL,
        )

    def test_fallback_can_be_disabled_for_samples(self):
        """Sample failures should not invoke the expensive local full-file API."""
        with (
            patch.object(
                shazam_year,
                "fetch_shazam_data",
                side_effect=RuntimeError("songrec failed"),
            ),
            patch.object(shazam_year, "fetch_local_shazam_data") as local,
        ):
            with self.assertRaisesRegex(RuntimeError, "songrec failed"):
                shazam_year.recognize_audio("sample.wav", local_fallback=False)

        local.assert_not_called()

    def test_similar_tracks_are_normalized(self):
        """Related-track API data should use the same normalization rules."""
        response = Mock()
        response.json.return_value = {
            "tracks": [
                {
                    "subtitle": "Artist {MIXED}",
                    "title": "Song (Live)",
                }
            ]
        }

        with patch.object(shazam_year.requests, "get", return_value=response) as get:
            result = shazam_year.fetch_similar_tracks(_shazam_result())

        self.assertEqual(result, [{"artist": "Artist", "title": "Song"}])
        response.raise_for_status.assert_called_once_with()
        self.assertEqual(get.call_args.kwargs["timeout"], shazam_year.DEFAULT_TIMEOUT)

    def test_query_shazam_uses_earliest_supported_year(self):
        """The final year should be the earliest Shazam or MusicBrainz date."""
        year_data = {
            "reldate": datetime(1997, 1, 1),
            "recdate": datetime(1998, 1, 1),
        }

        with (
            patch.object(shazam_year, "recognize_audio", return_value=_shazam_result(year="1999")),
            patch.object(shazam_year, "_query_recordings", return_value=year_data),
        ):
            result = shazam_year.query_shazam("song.mp3")

        self.assertEqual(result["artist"], "The Artist")
        self.assertEqual(result["title"], "The Song")
        self.assertEqual(result["year"], 1997)

    def test_year_only_uses_filename_without_recognizing_audio(self):
        """Year-only mode should skip Shazam and query parsed filename text."""
        year_data = {
            "reldate": datetime(1971, 1, 1),
            "recdate": datetime(1972, 1, 1),
        }

        with (
            patch.object(shazam_year, "recognize_audio") as recognize,
            patch.object(
                shazam_year,
                "_query_recordings",
                return_value=year_data,
            ) as query,
        ):
            result = shazam_year.query_shazam(
                "The Artist - The Song (1999).mp3",
                year_only=True,
            )

        recognize.assert_not_called()
        query.assert_called_once_with(
            artist="The Artist",
            title="The Song",
            first_only=False,
            result_limit=shazam_year.DEFAULT_RESULT_LIMIT,
        )
        self.assertEqual(result["year"], 1971)


class SamplingTests(unittest.TestCase):
    """Verify sample placement, extraction cleanup, and consensus."""

    def test_sample_start_handles_short_tracks_and_uniform_selection(self):
        """Sampling must remain within the available track duration."""
        self.assertEqual(shazam_year.generate_sample_start("uniform", 20, 30), 0.0)

        with patch.object(shazam_year.random, "uniform", return_value=42) as uniform:
            self.assertEqual(
                shazam_year.generate_sample_start("uniform", 100, 10),
                42,
            )
        uniform.assert_called_once_with(0, 90)

    def test_biased_sample_handles_zero_random_value_and_stays_in_bounds(self):
        """The Box-Muller transform must not evaluate log(0)."""
        with patch.object(
            shazam_year.random,
            "random",
            side_effect=[0.0, 0.5],
        ):
            start = shazam_year.generate_sample_start("biased", 100, 10)

        self.assertTrue(math.isfinite(start))
        self.assertGreaterEqual(start, 0)
        self.assertLessEqual(start, 90)

    def test_unknown_sample_method_is_rejected(self):
        """Programmatic callers should receive clear sampling validation."""
        with self.assertRaisesRegex(ValueError, "Unknown sampling method"):
            shazam_year.generate_sample_start("mystery", 100, 10)

    def test_failed_extraction_removes_temporary_file(self):
        """A failed SoX invocation must not leak its destination file."""
        with patch.object(
            shazam_year.sh,
            "sox",
            side_effect=RuntimeError("sox failed"),
        ) as sox:
            with self.assertRaisesRegex(RuntimeError, "sox failed"):
                shazam_year.extract_sample("song.mp3", 10, 30)

        temp_path = Path(sox.call_args.args[1])
        self.assertFalse(temp_path.exists())

    def test_non_consensus_returns_after_first_success_and_cleans_sample(self):
        """First-match mode should avoid unnecessary recognition requests."""
        sample_path = "/tmp/shazam-year-first-success.wav"
        first = {"artist": "A", "title": "Song", "year": 2001, "raw": {}}

        with (
            patch.object(shazam_year, "get_song_duration", return_value=120),
            patch.object(shazam_year, "generate_sample_start", return_value=10),
            patch.object(shazam_year, "extract_sample", return_value=sample_path),
            patch.object(shazam_year, "recognize_sample", return_value=first) as recognize,
            patch.object(Path, "unlink_p") as unlink,
        ):
            result = shazam_year.recognize_song_samples(
                "song.mp3",
                num_samples=5,
                sample_duration=30,
                consensus=False,
            )

        self.assertIs(result, first)
        recognize.assert_called_once()
        unlink.assert_called_once_with()

    def test_consensus_returns_winner_with_earliest_recognized_year(self):
        """Consensus should combine matching results and retain the oldest year."""
        results = [
            {"artist": "A", "title": "Song", "year": 2001, "raw": {"id": 1}},
            {"artist": "A", "title": "Song", "year": 1999, "raw": {"id": 2}},
        ]

        with (
            patch.object(shazam_year, "get_song_duration", return_value=120),
            patch.object(shazam_year, "generate_sample_start", side_effect=[10, 40]),
            patch.object(shazam_year, "extract_sample", side_effect=["one.wav", "two.wav"]),
            patch.object(shazam_year, "recognize_sample", side_effect=results) as recognize,
            patch.object(Path, "unlink_p") as unlink,
        ):
            result = shazam_year.recognize_song_samples(
                "song.mp3",
                num_samples=3,
                sample_duration=30,
                consensus=True,
            )

        self.assertEqual(result["artist"], "A")
        self.assertEqual(result["title"], "Song")
        self.assertEqual(result["year"], 1999)
        self.assertEqual(result["raw"], {"id": 1})
        self.assertEqual(recognize.call_count, 2)
        self.assertEqual(unlink.call_count, 2)

    def test_sampling_validates_count_duration_and_recognition_failure(self):
        """Invalid or unsuccessful sampling should fail with useful messages."""
        with self.assertRaisesRegex(ValueError, "num_samples must be at least 1"):
            shazam_year.recognize_song_samples("song.mp3", num_samples=0)

        with patch.object(shazam_year, "get_song_duration", return_value=10):
            with self.assertRaisesRegex(ValueError, "Track is too short"):
                shazam_year.recognize_song_samples(
                    "song.mp3",
                    num_samples=2,
                    sample_duration=30,
                )

        with (
            patch.object(shazam_year, "get_song_duration", return_value=120),
            patch.object(shazam_year, "generate_sample_start", return_value=10),
            patch.object(shazam_year, "extract_sample", return_value="sample.wav"),
            patch.object(shazam_year, "recognize_sample", return_value=None),
            patch.object(Path, "unlink_p"),
        ):
            with self.assertRaisesRegex(RuntimeError, "No sampled segment"):
                shazam_year.recognize_song_samples(
                    "song.mp3",
                    num_samples=2,
                    sample_duration=30,
                )


class FileAndCommandLineTests(unittest.TestCase):
    """Verify safe renaming and command-line parsing."""

    def test_rename_song_creates_canonical_destination(self):
        """Renaming should preserve the source extension and content."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.flac"
            source.write_bytes(b"audio")

            shazam_year.rename_song(
                source,
                {"artist": "The Cure", "title": "Love Song", "year": 1989},
            )

            destination = Path(directory) / "Cure - Love Song (1989).flac"
            self.assertFalse(source.exists())
            self.assertEqual(destination.bytes(), b"audio")

    def test_parser_requires_song_and_validates_sample_method(self):
        """Argparse should enforce required inputs and supported strategies."""
        parser = shazam_year.build_argument_parser()

        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parser.parse_args([])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--sample-method", "unknown", "song.mp3"])

        args = parser.parse_args(
            ["--num-samples", "3", "--sample-method", "biased", "song.mp3"]
        )
        self.assertEqual(args.num_samples, 3)
        self.assertEqual(args.sample_method, "biased")
        self.assertEqual([str(path) for path in args.song_paths], ["song.mp3"])


if __name__ == "__main__":
    unittest.main()
