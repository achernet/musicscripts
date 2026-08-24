"""Regression tests for YouTube candidate retrieval and ranking."""

import unittest
from unittest.mock import patch

import numpy as np
from yt_dlp.utils import DownloadError

from youtube_scorer import (
    build_search_query,
    extract_description_penalty,
    extract_keyword_score,
    fetch_youtube_results,
    get_artist_match_score,
    get_title_match_score,
    score_youtube_results,
    smart_duration_clusters_pure,
    sort_scored_results,
)


class TitleMatchScoreTests(unittest.TestCase):
    """Verify title normalization and confidence floors."""

    QUERY = "whole lot of shakin goin on"

    def test_punctuation_and_spacing_variants_are_exact_matches(self):
        """Formatting differences must not reduce an otherwise exact match."""
        variants = [
            "Whole Lot of Shakin' Goin' On",
            "Whole Lot of Shakin Goin On",
            'Jerry Lee Lewis & Kid Rock - "Whole Lot of Shakin´ Goin´On"',
        ]

        for title in variants:
            with self.subTest(title=title):
                self.assertEqual(get_title_match_score(self.QUERY, title), 1.0)

    def test_unrelated_title_scores_zero(self):
        """Incidental fuzzy overlap must remain below the confidence floor."""
        self.assertEqual(
            get_title_match_score(self.QUERY, "Completely Different Song"),
            0.0,
        )

    def test_empty_title_scores_zero(self):
        """Missing candidate titles must be safe and contribute nothing."""
        self.assertEqual(get_title_match_score(self.QUERY, ""), 0.0)

    def test_colloquial_and_standard_spellings_are_equivalent(self):
        """Common rock-and-roll spellings should normalize to one title."""
        self.assertEqual(
            get_title_match_score(
                "whole lotta shakin goin on",
                "Whole Lot Of Shaking Going On",
            ),
            1.0,
        )


class ArtistMatchScoreTests(unittest.TestCase):
    """Verify artist matching against channel and title metadata."""

    def test_exact_artist_in_channel_or_title_scores_one(self):
        """An exact artist in either source should receive full credit."""
        self.assertEqual(
            get_artist_match_score(
                ["Jerry Lee Lewis"],
                "Jerry Lee Lewis",
                "Whole Lot of Shakin' Goin' On",
            ),
            1.0,
        )
        self.assertEqual(
            get_artist_match_score(
                ["Jerry Lee Lewis"],
                "Random Channel",
                "Jerry Lee Lewis - Whole Lot of Shakin' Goin' On",
            ),
            1.0,
        )

    def test_unrelated_artist_scores_zero(self):
        """Unrelated uploader and title metadata must receive no credit."""
        self.assertEqual(
            get_artist_match_score(
                ["Jerry Lee Lewis"],
                "Random Channel",
                "Completely Different Song",
            ),
            0.0,
        )

    def test_channel_name_without_spaces_is_an_exact_artist_match(self):
        """Concatenated official channel names should retain full artist credit."""
        self.assertEqual(
            get_artist_match_score(
                ["Jerry Lee Lewis"],
                "JerryLeeLewisTV",
                "Whole Lotta Shakin' Goin' On",
            ),
            1.0,
        )


class KeywordScoreTests(unittest.TestCase):
    """Verify positive preferences and negative version signals."""

    def test_live_and_karaoke_are_heavily_penalized(self):
        """Undesired alternate versions should receive strong penalties."""
        self.assertEqual(extract_keyword_score("Song (Live)")[0], -0.50)
        self.assertEqual(extract_keyword_score("Song (Karaoke Version)")[0], -0.75)

    def test_keywords_only_match_whole_words(self):
        """Keyword fragments inside unrelated words must not match."""
        self.assertEqual(extract_keyword_score("Deliver Me")[0], 0.0)

    def test_remastered_receives_remaster_bonus(self):
        """The remaster stem should recognize the common past-tense form."""
        self.assertEqual(extract_keyword_score("Official Remastered Audio")[0], 0.55)

    def test_show_is_penalized_and_lyric_uploads_are_boosted(self):
        """Talk-show risk should lose ground to direct lyric uploads."""
        self.assertEqual(extract_keyword_score("Song (Steve Allen Show)")[0], -1.00)
        self.assertEqual(extract_keyword_score("Song (Lyrics)")[0], 0.15)
        self.assertEqual(extract_keyword_score("Song(Lyrics)")[0], 0.15)
        self.assertAlmostEqual(extract_keyword_score("Song (Lyric Video)")[0], 0.10)

    def test_negative_keyword_in_query_is_not_penalized(self):
        """A legitimate song or artist name must protect its own keywords."""
        score, matches = extract_keyword_score(
            "The Show Must Go On (Official Video)",
            protected_texts=["The Show Must Go On", "Queen"],
        )
        self.assertEqual(score, 0.20)
        self.assertNotIn(("show", -1.00), matches)

    def test_movie_film_and_dvd_sources_are_penalized(self):
        """Narrative-video sources should rank below clean audio uploads."""
        self.assertEqual(extract_keyword_score("Song From the Movie")[0], -0.75)
        self.assertEqual(extract_keyword_score("Song - Película Completa")[0], -0.75)
        self.assertEqual(extract_keyword_score("Song from a Film")[0], -0.50)
        self.assertEqual(extract_keyword_score("Song from DVD")[0], -0.35)

    def test_tutorial_and_performance_signals_are_penalized(self):
        """Lessons and documented performances should not beat clean tracks."""
        self.assertEqual(extract_keyword_score("How to Play Song - Piano Tutorial")[0], -1.00)
        self.assertEqual(extract_keyword_score("Song - Piano Lesson")[0], -0.80)
        self.assertEqual(extract_keyword_score("Song in Concert")[0], -0.50)
        self.assertEqual(extract_keyword_score("Song on Stage")[0], -0.35)

    def test_performance_word_in_query_is_not_penalized(self):
        """Keywords remain valid when they are part of the requested song."""
        score, matches = extract_keyword_score(
            "Concert for Aliens (Official Audio)",
            protected_texts=["Concert for Aliens", "Machine Gun Kelly"],
        )
        self.assertEqual(score, 0.35)
        self.assertNotIn(("concert", -0.50), matches)

    def test_description_adds_only_unique_negative_signals(self):
        """Descriptions should reveal live versions without double penalties."""
        score, matches = extract_description_penalty(
            "Acoustic Alchemy live at St. Lucia, 2003. Official upload.",
            protected_texts=["Angel of the South", "Acoustic Alchemy"],
        )
        self.assertEqual(score, -0.50)
        self.assertEqual(matches, [("description: live", -0.50)])

        duplicate_score, duplicate_matches = extract_description_penalty(
            "Recorded live in concert.",
            excluded_keywords=["live"],
        )
        self.assertEqual(duplicate_score, -0.50)
        self.assertEqual(duplicate_matches, [("description: concert", -0.50)])


class ResultScoringTests(unittest.TestCase):
    """Verify complete-result scoring and ordering behavior."""

    def test_missing_metadata_is_scored_safely(self):
        """Missing duration and a null view count must not abort scoring."""
        scored = score_youtube_results(
            [
                {
                    "title": "Whole Lot of Shakin' Goin' On",
                    "uploader": "Jerry Lee Lewis",
                    "view_count": None,
                }
            ],
            "whole lot of shakin goin on",
            ["Jerry Lee Lewis"],
        )

        self.assertEqual(len(scored), 1)
        self.assertEqual(scored[0]["log_views"], 0.0)
        self.assertEqual(scored[0]["duration_score"], 0.0)
        self.assertIn("final_score", scored[0])

    def test_live_description_penalizes_an_unmarked_title(self):
        """A live description must affect scoring when the title omits it."""
        scored = score_youtube_results(
            [
                {
                    "id": "18qPj64d2Cw",
                    "title": "Acoustic Alchemy-Angel of the South",
                    "description": "Acoustic Alchemy live at St. Lucia, 2003.",
                    "uploader": "Acoustic Alchemy",
                    "duration": 426,
                    "view_count": 195269,
                }
            ],
            "Angel of the South",
            ["Acoustic Alchemy"],
        )

        self.assertEqual(scored[0]["keyword_boost"], -0.50)
        self.assertIn(
            ("description: live", -0.50),
            scored[0]["matched_keywords"],
        )

    def test_strong_match_outranks_unrelated_first_result(self):
        """Local match quality should overcome retrieval-position advantage."""
        scored = score_youtube_results(
            [
                {
                    "id": "wrong",
                    "title": "Completely Different Song",
                    "uploader": "Random Channel",
                    "duration": 180,
                    "view_count": 100,
                },
                {
                    "id": "right",
                    "title": "Jerry Lee Lewis - Whole Lot of Shakin' Goin' On",
                    "uploader": "Jerry Lee Lewis",
                    "duration": 180,
                    "view_count": 100,
                },
            ],
            "whole lot of shakin goin on",
            ["Jerry Lee Lewis"],
        )

        self.assertEqual(scored[0]["id"], "right")

    def test_clean_official_upload_outranks_movie_and_show_versions(self):
        """The problematic Jerry Lee Lewis fixtures should favor clean audio."""
        scored = score_youtube_results(
            [
                {
                    "id": "JQhQq0Pg_V8",
                    "title": "Whole Lotta Shakin' Goin' On Jerry Lee Lewis From the Movie",
                    "uploader": "Sam the toast",
                    "duration": 269,
                    "view_count": 1_548_881,
                },
                {
                    "id": "GN8VV8CHnrk",
                    "title": "Whole Lotta Shakin' Goin' On",
                    "uploader": "JerryLeeLewisTV",
                    "duration": 177,
                    "view_count": 12_016_287,
                },
                {
                    "id": "Fw7SBF-35Es",
                    "title": "Jerry Lee Lewis - Whole Lotta Shakin' Goin' On (Steve Allen Show)",
                    "uploader": "John1948SIxC",
                    "duration": 174,
                    "view_count": 6_351_992,
                },
            ],
            "whole lotta shakin goin on",
            ["Jerry Lee Lewis"],
        )

        self.assertEqual(scored[0]["id"], "GN8VV8CHnrk")
        self.assertEqual(scored[0]["artist_match_score"], 1.0)
        show_result = next(
            item for item in scored if item["id"] == "Fw7SBF-35Es"
        )
        self.assertLess(show_result["final_score"], scored[0]["final_score"])

    def test_explicit_sort_direction_is_not_inverted(self):
        """Descending and ascending requests should be semantically direct."""
        scored = score_youtube_results(
            [
                {"id": "a", "title": "A", "duration": 180},
                {"id": "b", "title": "B", "duration": 180},
            ],
            "A",
            ["Artist"],
        )

        descending = sort_scored_results(scored, descending=True)
        ascending = sort_scored_results(scored, descending=False)
        self.assertGreaterEqual(descending[0]["final_score"], descending[-1]["final_score"])
        self.assertLessEqual(ascending[0]["final_score"], ascending[-1]["final_score"])


class DurationClusterTests(unittest.TestCase):
    """Verify duration clustering fallbacks and common modes."""

    def test_empty_durations_have_a_stable_fallback(self):
        """An empty sample should still yield a safe default cluster."""
        self.assertEqual(smart_duration_clusters_pure(np.array([])), [(240.0, 60.0, 0)])

    def test_dense_duration_family_produces_a_nearby_cluster(self):
        """A tight family of song lengths should produce a nearby center."""
        clusters = smart_duration_clusters_pure(
            np.array([198, 199, 200, 201, 202, 203, 204], dtype=float)
        )
        self.assertLess(abs(clusters[0][0] - 201), 5)


class RetrievalTests(unittest.TestCase):
    """Verify query construction and yt-dlp boundary behavior."""

    def test_search_query_contains_only_artist_and_title(self):
        """Ranking preferences must never narrow YouTube retrieval."""
        query = build_search_query("Playing for Time", ["Acoustic Alchemy"], 30)
        self.assertEqual(
            query,
            "ytsearch30:('Acoustic Alchemy') 'Playing for Time'",
        )
        self.assertNotIn("official", query)

    @patch("youtube_scorer.yt_dlp.YoutubeDL")
    def test_download_error_returns_empty_results(self, youtube_dl):
        """A yt-dlp failure should be reported without an uncaught exception."""
        downloader = youtube_dl.return_value.__enter__.return_value
        downloader.extract_info.side_effect = DownloadError("network failure")

        self.assertEqual(fetch_youtube_results("Song", ["Artist"]), [])

    @patch("youtube_scorer.yt_dlp.YoutubeDL")
    def test_results_without_duration_are_preserved(self, youtube_dl):
        """Incomplete search entries should remain available for other signals."""
        downloader = youtube_dl.return_value.__enter__.return_value
        downloader.extract_info.return_value = {
            "entries": [
                {"id": "missing", "title": "Song", "duration": None},
                {"id": "known", "title": "Song", "duration": 180},
            ]
        }

        results = fetch_youtube_results("Song", ["Artist"])
        self.assertEqual([result["id"] for result in results], ["missing", "known"])
        self.assertNotIn("duration_string", results[0])
        self.assertEqual(results[1]["duration_string"], "3:00")


if __name__ == "__main__":
    unittest.main()
