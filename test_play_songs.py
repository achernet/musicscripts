"""Tests for playlist selection, ordering, and player lifecycle helpers."""

import math
import subprocess
import unittest
from contextlib import redirect_stderr
from dataclasses import dataclass
from io import StringIO
from unittest.mock import Mock, call, patch

import play_songs
from play_songs import RandomSorter


@dataclass(frozen=True)
class FakeSong:
    """Minimal path-like object exposing timestamps used by RandomSorter."""

    name: str
    atime: float
    mtime: float
    ctime: float = 0.0

    def basename(self):
        """Return the filename, matching path.Path's API."""
        return self.name


class RandomSorterTests(unittest.TestCase):
    """Verify deterministic and randomized playlist ordering."""

    def test_explicit_false_and_zero_values_override_defaults(self):
        """Falsy caller settings must not be replaced by class defaults."""
        sorter = RandomSorter(
            use_atime=False,
            use_mtime=False,
            factor=0,
            offset=0,
            reverse=False,
        )

        self.assertFalse(sorter.use_atime)
        self.assertFalse(sorter.use_mtime)
        self.assertEqual(sorter.factor, 0)
        self.assertEqual(sorter.offset, 0)
        self.assertFalse(sorter.reverse)

    def test_plain_random_mode_shuffles_a_copy(self):
        """Random mode should not mutate the caller's collection."""
        songs = ["one", "two", "three"]

        with patch.object(play_songs.random, "shuffle", side_effect=lambda values: values.reverse()):
            result = RandomSorter().random_sort_by_time(songs)

        self.assertEqual(result, ["three", "two", "one"])
        self.assertEqual(songs, ["one", "two", "three"])

    def test_atime_sort_uses_access_time_not_change_time(self):
        """The --atime behavior must read the actual access timestamp."""
        song = FakeSong("song.mp3", atime=99, mtime=20, ctime=1)

        with patch.object(play_songs.time, "time", return_value=100):
            key = RandomSorter(use_atime=True).sort_key(song)

        self.assertAlmostEqual(key, math.log1p(1))

    def test_equal_timestamp_is_safe(self):
        """A timestamp equal to the current instant must not cause log(0)."""
        song = FakeSong("song.mp3", atime=100, mtime=100)

        with patch.object(play_songs.time, "time", return_value=100):
            self.assertEqual(RandomSorter(use_mtime=True).sort_key(song), 0.0)

    def test_time_sort_can_put_recent_songs_first_or_last(self):
        """Reverse should directly control the timestamp preference."""
        recent = FakeSong("recent.mp3", atime=99, mtime=99)
        old = FakeSong("old.mp3", atime=1, mtime=1)

        with patch.object(play_songs.time, "time", return_value=100):
            forward = RandomSorter(use_mtime=True, factor=0, offset=1)
            reverse = RandomSorter(use_mtime=True, factor=0, offset=1, reverse=True)
            self.assertEqual(forward.random_sort_by_time([old, recent]), [recent, old])
            self.assertEqual(reverse.random_sort_by_time([old, recent]), [old, recent])

    def test_weighted_sort_preserves_duplicate_entries(self):
        """Repeated playlist entries should not disappear during weighting."""
        song = FakeSong("repeat.mp3", atime=99, mtime=99)
        sorter = RandomSorter(use_mtime=True, factor=0, offset=1)

        with patch.object(play_songs.time, "time", return_value=100):
            result = sorter.random_sort_by_time([song, song])

        self.assertEqual(result, [song, song])


class PlaylistSelectionTests(unittest.TestCase):
    """Verify filtering and playlist option precedence."""

    def test_songs_matching_is_case_insensitive_and_can_exclude(self):
        """Filename matching should support both allowlists and denylists."""
        songs = [
            FakeSong("Miles Davis.mp3", 0, 0),
            FakeSong("Rock Song.mp3", 0, 0),
        ]

        with patch.object(play_songs, "list_all_songs", return_value=songs):
            self.assertEqual(play_songs.songs_matching("miles"), [songs[0]])
            self.assertEqual(play_songs.songs_matching("miles", include=False), [songs[1]])

    def test_trance_random_takes_priority_over_other_playlist_options(self):
        """The documented historical option priority should remain stable."""
        sorter = Mock()
        sorter.random_sort_by_time.side_effect = lambda songs: list(songs)

        with patch.object(play_songs, "songs_matching", return_value=["trance.mp3"]) as matching:
            result = play_songs.gather_songs(
                sorter,
                old=True,
                trance=True,
                classical=True,
            )

        self.assertEqual(result, ["trance.mp3"])
        matching.assert_called_once()

    def test_jazz_combines_random_and_folder_sources(self):
        """Both jazz bit flags should contribute songs to one playlist."""
        sorter = Mock()
        sorter.random_sort_by_time.side_effect = lambda songs: list(songs)
        jazz_path = Mock()
        jazz_path.walkfiles.return_value = ["folder.mp3"]

        with (
            patch.object(play_songs, "songs_matching", return_value=["random.mp3"]),
            patch.object(play_songs, "JAZZ_PATH", jazz_path),
        ):
            result = play_songs.gather_songs(
                sorter,
                jazz=play_songs.JAZZ_RANDOM_BIT | play_songs.JAZZ_FOLDER_BIT,
            )

        self.assertEqual(result, ["random.mp3", "folder.mp3"])
        jazz_path.walkfiles.assert_called_once_with("*.mp3")

    def test_explicit_songs_are_extended_only_when_requested(self):
        """Explicit files play alone unless a continuation option is active."""
        sorter = Mock(use_atime=False, use_mtime=False)
        sorter.random_sort_by_time.side_effect = lambda songs: list(songs)

        with patch.object(play_songs, "gather_songs", return_value=["library.mp3"]) as gather:
            songs_only = play_songs.play_songs(
                sorter,
                songs=["chosen.mp3"],
                list_only=True,
            )
            continued = play_songs.play_songs(
                sorter,
                songs=["chosen.mp3"],
                list_only=True,
                continue_play=True,
            )

        self.assertEqual(songs_only, ["chosen.mp3"])
        self.assertEqual(continued, ["chosen.mp3", "library.mp3"])
        gather.assert_called_once()


class PlayerLifecycleTests(unittest.TestCase):
    """Verify failures and process cleanup without launching real audio."""

    def test_missing_sox_player_has_a_clear_error(self):
        """A missing external player should fail before creating a process."""
        with patch.object(play_songs, "PLAY_PATH", None):
            with self.assertRaisesRegex(RuntimeError, "SoX 'play' executable"):
                play_songs.play_song_showing_console_output("song.mp3")

    def test_stop_player_interrupts_and_reaps_unix_process_group(self):
        """Stopping playback should signal the whole group and wait for it."""
        proc = Mock(pid=123)
        proc.poll.return_value = None

        with (
            patch.object(play_songs.os, "name", "posix"),
            patch.object(play_songs.os, "killpg") as killpg,
        ):
            play_songs.stop_player(proc)

        killpg.assert_called_once_with(123, play_songs.signal.SIGINT)
        proc.wait.assert_called_once_with(timeout=2)

    def test_stop_player_kills_process_that_ignores_interrupt(self):
        """A player that misses the graceful deadline should be force-killed."""
        proc = Mock(pid=123)
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired("play", 2), None]

        with (
            patch.object(play_songs.os, "name", "posix"),
            patch.object(play_songs.os, "killpg"),
        ):
            play_songs.stop_player(proc)

        proc.kill.assert_called_once_with()
        self.assertEqual(proc.wait.call_args_list, [call(timeout=2), call()])


class CommandLineTests(unittest.TestCase):
    """Verify parser-level validation and defaults."""

    def test_trance_modes_are_mutually_exclusive(self):
        """Conflicting trance options should produce normal argparse usage."""
        parser = play_songs.build_argument_parser()

        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parser.parse_args(["--trance-random", "--trance-users"])

        self.assertEqual(raised.exception.code, 2)

    def test_cli_random_settings_are_parsed_as_floats(self):
        """Numeric tuning options should reach RandomSorter as numbers."""
        args = play_songs.build_argument_parser().parse_args(
            ["--random-factor", "0", "--random-offset", "1", "song.mp3"]
        )

        self.assertEqual(args.random_factor, 0.0)
        self.assertEqual(args.random_offset, 1.0)
        self.assertEqual([str(song) for song in args.songs], ["song.mp3"])


if __name__ == "__main__":
    unittest.main()
