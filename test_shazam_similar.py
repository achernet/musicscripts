"""Offline tests for Shazam related-track discovery and YouTube selection."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

from path import Path

import shazam_similar


class EmbeddedJsonTests(unittest.TestCase):
    """Verify extraction of Shazam's JSON embedded in JavaScript strings."""

    def test_extracts_object_and_array_from_quoted_javascript(self):
        """Escaped JSON can contain surrounding non-JSON text."""
        object_element = SimpleNamespace(
            text='window.data = "{\\"url\\": \\"https://example.test\\", \\"citation\\": []}";'
        )
        array_element = SimpleNamespace(text='prefix "before [1, 2, 3] after" suffix')

        self.assertEqual(
            shazam_similar.extract_json_from_element(object_element),
            {"url": "https://example.test", "citation": []},
        )
        self.assertEqual(
            shazam_similar.extract_json_from_element(array_element),
            [1, 2, 3],
        )

    def test_missing_quoted_json_has_a_clear_error(self):
        """Malformed page data should not leak StopIteration or TypeError."""
        for text in (None, "no quoted data"):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "quoted JSON string"):
                    shazam_similar.extract_json_from_element(
                        SimpleNamespace(text=text)
                    )

    def test_can_extract_json_is_a_safe_predicate(self):
        """The predicate should return False for malformed embedded data."""
        good = SimpleNamespace(text='prefix "[1]" suffix')
        bad = SimpleNamespace(text="not json")

        self.assertTrue(shazam_similar.can_extract_json(good))
        self.assertFalse(shazam_similar.can_extract_json(bad))


class SongParsingTests(unittest.TestCase):
    """Verify filename and Shazam-field normalization."""

    def test_normalize_query_removes_leading_the_and_nested_annotations(self):
        """Version metadata should not become part of search queries."""
        self.assertEqual(
            shazam_similar.normalize_query(
                "The Song (Live [Fillmore]) {Remastered}"
            ),
            "Song",
        )

    def test_parse_song_parts_splits_and_deduplicates_artists(self):
        """Ampersand/comma collaborators should become unique artist terms."""
        result = shazam_similar.parse_song_parts(
            "The Alpha & Beta, Alpha",
            "The Song (Live)",
        )

        self.assertEqual(
            result,
            {"artists": ["Alpha", "Beta"], "title": "Song", "file": None},
        )

    def test_parse_filename_removes_directory_year_and_bracket_junk(self):
        """Canonical MP3 filenames should produce clean search fields."""
        result = shazam_similar.parse_song_filename(
            "/music/The Alpha & Beta - The Song [Mix] (2001).MP3"
        )

        self.assertEqual(result["artists"], ["Alpha", "Beta"])
        self.assertEqual(result["title"], "Song")
        self.assertEqual(result["file"], Path("The Alpha & Beta - The Song [Mix] (2001).MP3"))

    def test_parse_filename_rejects_missing_separator(self):
        """Invalid filenames should identify the expected separator."""
        with self.assertRaisesRegex(ValueError, "missing ' - ' separator"):
            shazam_similar.parse_song_filename("Artist Song.mp3")


class InvokeShazamTests(unittest.TestCase):
    """Verify recognition and Shazam-page parsing boundaries."""

    def test_get_shazam_url_uses_consensus_sampling(self):
        """Related lookup should recognize the source with configured sampling."""
        recognized = {"raw": {"track": {"url": "https://shazam.test/track"}}}

        with patch.object(
            shazam_similar,
            "recognize_song_samples",
            return_value=recognized,
        ) as recognize:
            result = shazam_similar.get_shazam_url("song.mp3")

        self.assertEqual(result, "https://shazam.test/track")
        recognize.assert_called_once_with(
            "song.mp3",
            num_samples=shazam_similar.SHAZAM_NUM_SAMPLES,
            sample_duration=shazam_similar.SHAZAM_SAMPLE_DURATION,
            consensus=True,
        )

    def test_get_shazam_url_rejects_missing_or_empty_url(self):
        """Incomplete recognition data should produce a domain-level error."""
        for recognized in ({"raw": None}, {"raw": {"track": {"url": ""}}}):
            with self.subTest(recognized=recognized):
                with patch.object(
                    shazam_similar,
                    "recognize_song_samples",
                    return_value=recognized,
                ):
                    with self.assertRaisesRegex(RuntimeError, "track URL"):
                        shazam_similar.get_shazam_url("song.mp3")

    def test_title_with_slash_is_not_treated_as_a_file_path(self):
        """A slash inside a Shazam title must remain title text."""
        shazam_url = "https://www.shazam.com/track/49484934/incense-and-peppermints"
        citation = {
            "url": shazam_url,
            "citation": [
                {
                    "@type": "MusicRecording",
                    "byArtist": "Joe Cocker",
                    "name": "The Letter (Live At The Fillmore East/1970)",
                }
            ],
        }

        response = Mock(text="<html></html>", url=shazam_url)
        tree = Mock()
        tree.xpath.return_value = [object()]

        with (
            patch.object(shazam_similar, "get_shazam_url", return_value=shazam_url),
            patch.object(shazam_similar.requests, "get", return_value=response),
            patch.object(shazam_similar.etree, "fromstring", return_value=tree),
            patch.object(shazam_similar, "extract_json_from_element", return_value=citation),
            patch.object(shazam_similar, "already_downloaded_path", return_value=None),
        ):
            songs = shazam_similar.invoke_shazam("song.mp3")

        self.assertEqual(
            songs,
            [
                {
                    "artists": ["Joe Cocker"],
                    "title": "Letter",
                    "file": None,
                    "match": "similar",
                }
            ],
        )

    def test_http_timeout_and_missing_metadata_are_handled(self):
        """Shazam page requests need a timeout and clear schema errors."""
        shazam_url = "https://www.shazam.com/track/123/song"
        response = Mock(text="<html></html>", url=shazam_url)
        tree = Mock()
        tree.xpath.return_value = []

        with (
            patch.object(shazam_similar, "get_shazam_url", return_value=shazam_url),
            patch.object(shazam_similar.requests, "get", return_value=response) as get,
            patch.object(shazam_similar.etree, "fromstring", return_value=tree),
        ):
            with self.assertRaisesRegex(RuntimeError, "related-song metadata"):
                shazam_similar.invoke_shazam("song.mp3")

        get.assert_called_once_with(
            shazam_url,
            headers=shazam_similar.HTTP_HEADERS,
            timeout=shazam_similar.HTTP_TIMEOUT,
        )
        tree.make_links_absolute.assert_called_once_with(shazam_url)

    def test_duplicate_album_citations_are_returned_once(self):
        """Duplicate Shazam citations should not repeat recommendations."""
        shazam_url = "https://www.shazam.com/track/123/song"
        related = {
            "@type": "MusicRecording",
            "byArtist": "The Artist",
            "name": "The Song",
            "inAlbum": "Album",
        }
        citation = {"url": shazam_url, "citation": [related, dict(related)]}
        response = Mock(text="<html></html>", url=shazam_url)
        tree = Mock()
        tree.xpath.return_value = [object()]

        with (
            patch.object(shazam_similar, "get_shazam_url", return_value=shazam_url),
            patch.object(shazam_similar.requests, "get", return_value=response),
            patch.object(shazam_similar.etree, "fromstring", return_value=tree),
            patch.object(shazam_similar, "extract_json_from_element", return_value=citation),
            patch.object(shazam_similar, "already_downloaded_path", return_value=None),
        ):
            songs = shazam_similar.invoke_shazam("song.mp3")

        self.assertEqual(
            songs,
            [
                {
                    "artists": ["Artist"],
                    "title": "Song",
                    "file": None,
                    "match": "artist",
                }
            ],
        )


class DownloadedLibraryTests(unittest.TestCase):
    """Verify matching recommendations against the local library index."""

    def test_matching_title_and_any_artist_returns_downloaded_path(self):
        """One shared artist is sufficient when normalized titles match."""
        downloaded = Path("Alpha & Beta - Love - Song.mp3")
        index = {
            "love - song": [
                {
                    "artists": ["Alpha", "Beta"],
                    "title": "Love - Song",
                    "file": downloaded,
                }
            ]
        }
        song = {"artists": ["Other", "beta"], "title": "Love/Song"}

        self.assertEqual(
            shazam_similar.already_downloaded_path(song, index),
            downloaded,
        )

    def test_title_or_artist_mismatch_returns_none(self):
        """A matching title alone must not suppress another artist's song."""
        index = {
            "song": [
                {"artists": ["Alpha"], "title": "Song", "file": Path("a.mp3")}
            ]
        }

        self.assertIsNone(
            shazam_similar.already_downloaded_path(
                {"artists": ["Beta"], "title": "Song"},
                index,
            )
        )
        self.assertIsNone(
            shazam_similar.already_downloaded_path(
                {"artists": ["Alpha"], "title": "Other"},
                index,
            )
        )


class YouTubeResultTests(unittest.TestCase):
    """Verify URL handling, scoring summaries, and selected-result output."""

    def test_youtube_url_accepts_canonical_url_or_video_id(self):
        """yt-dlp result variants should map to one usable URL."""
        self.assertEqual(
            shazam_similar.youtube_result_url(
                {"webpage_url": "https://www.youtube.com/watch?v=abc"}
            ),
            "https://www.youtube.com/watch?v=abc",
        )
        self.assertEqual(
            shazam_similar.youtube_result_url({"id": "xyz"}),
            "https://www.youtube.com/watch?v=xyz",
        )
        self.assertEqual(
            shazam_similar.youtube_result_url({"url": "bare-id"}),
            "https://www.youtube.com/watch?v=bare-id",
        )

    def test_youtube_url_rejects_missing_identifier(self):
        """Missing result identity should not create a watch?v=None URL."""
        with self.assertRaisesRegex(ValueError, "URL or video ID"):
            shazam_similar.youtube_result_url({})

    def test_weighted_views_requires_both_artist_and_title_match(self):
        """View totals should be discounted by the joint match score."""
        results = [
            {
                "view_count": 100,
                "artist_match_score": 0.5,
                "title_boost": 0.8,
            },
            {
                "view_count": 10,
                "artist_match_score": 1.0,
                "title_boost": 1.0,
            },
            {
                "view_count": "invalid",
                "artist_match_score": 1.0,
                "title_boost": 1.0,
            },
        ]

        self.assertEqual(shazam_similar.total_match_weighted_views(results), 50)

    def test_display_contains_selection_metadata_and_mismatch_status(self):
        """Console output should make weak selections visible to the user."""
        song = {"artists": ["Artist"], "title": "Song"}
        result = {
            "id": "abc",
            "uploader": "Channel",
            "title": "Other Video",
            "view_count": 1234,
            "duration": 180,
            "artist_match_score": 0.4,
            "title_boost": 0.9,
        }

        output = StringIO()
        with redirect_stdout(output):
            shazam_similar.display_selected_youtube_result(
                song,
                result,
                [result],
            )

        rendered = output.getvalue()
        self.assertIn("Query:   Artist - Song", rendered)
        self.assertIn("LIKELY MISMATCH", rendered)
        self.assertIn("Selected URL views: 1,234", rendered)
        self.assertIn("https://www.youtube.com/watch?v=abc", rendered)


class WorkflowTests(unittest.TestCase):
    """Verify downloader construction and end-to-end recommendation flow."""

    def test_fetch_song_builds_expected_download_pipeline(self):
        """The download command should include audio conversion and recognition."""
        yt_dlp = Mock()
        command = Mock()
        baked = Mock()
        yt_dlp.bake.return_value = command
        command.bake.return_value = baked

        with (
            patch.object(shazam_similar.sh, "Command", return_value=yt_dlp) as command_factory,
            patch.object(
                shazam_similar,
                "run_command_showing_output",
                return_value="done",
            ) as run,
        ):
            result = shazam_similar.fetch_song("https://youtube.test/watch?v=abc")

        self.assertEqual(result, "done")
        command_factory.assert_called_once_with("yt-dlp")
        command.bake.assert_called_once_with("https://youtube.test/watch?v=abc")
        run.assert_called_once_with(baked)
        baked_args = yt_dlp.bake.call_args.args
        self.assertIn("--extract-audio", baked_args)
        self.assertIn("--exec", baked_args)
        self.assertTrue(any("shazam_year.py" in str(arg) for arg in baked_args))

    def test_download_workflow_skips_existing_song_and_fetches_best_new_result(self):
        """Only missing songs should reach YouTube and the downloader."""
        downloaded = {
            "artists": ["Old Artist"],
            "title": "Old Song",
            "file": Path("old.mp3"),
            "match": "artist",
        }
        missing = {
            "artists": ["New Artist"],
            "title": "New Song",
            "file": None,
            "match": "similar",
        }
        selected = {
            "id": "best",
            "title": "New Song",
            "artist_match_score": 1,
            "title_boost": 1,
        }
        index = {"old song": [downloaded]}

        with (
            patch.object(shazam_similar, "_build_downloaded_index", return_value=index),
            patch.object(
                shazam_similar,
                "invoke_shazam",
                return_value=[downloaded, missing],
            ) as invoke,
            patch.object(
                shazam_similar,
                "fetch_youtube_results",
                return_value=[{"id": "candidate"}],
            ) as fetch_results,
            patch.object(
                shazam_similar,
                "score_youtube_results",
                return_value=[selected],
            ) as score,
            patch.object(shazam_similar, "display_selected_youtube_result") as display,
            patch.object(shazam_similar, "fetch_song") as download,
            redirect_stdout(StringIO()),
        ):
            result = shazam_similar.fetch_similar_songs("source.mp3")

        self.assertEqual(result, [downloaded, missing])
        invoke.assert_called_once_with("source.mp3", index)
        fetch_results.assert_called_once_with(
            track_name="New Song",
            artists=["New Artist"],
            quantity=30,
        )
        score.assert_called_once_with(
            [{"id": "candidate"}],
            "New Song",
            ["New Artist"],
        )
        display.assert_called_once_with(missing, selected, [selected])
        download.assert_called_once_with("https://www.youtube.com/watch?v=best")

    def test_empty_scored_results_are_skipped(self):
        """No download or display should occur when ranking finds no candidate."""
        song = {
            "artists": ["Artist"],
            "title": "Song",
            "file": None,
            "match": "similar",
        }

        with (
            patch.object(shazam_similar, "_build_downloaded_index", return_value={}),
            patch.object(shazam_similar, "invoke_shazam", return_value=[song]),
            patch.object(shazam_similar, "fetch_youtube_results", return_value=[]),
            patch.object(shazam_similar, "score_youtube_results", return_value=[]),
            patch.object(shazam_similar, "display_selected_youtube_result") as display,
            patch.object(shazam_similar, "fetch_song") as download,
        ):
            shazam_similar.fetch_similar_songs("source.mp3")

        display.assert_not_called()
        download.assert_not_called()


class CommandLineTests(unittest.TestCase):
    """Verify command-line defaults and download mode."""

    def test_parser_defaults_to_list_only_behavior(self):
        """Downloads should require the explicit --download flag."""
        parser = shazam_similar.build_argument_parser()

        default = parser.parse_args(["song.mp3"])
        download = parser.parse_args(["--download", "song.mp3"])

        self.assertFalse(default.download)
        self.assertTrue(download.download)
        self.assertEqual(default.song_path, "song.mp3")


if __name__ == "__main__":
    unittest.main()
