#!/usr/bin/env python3
"""
Shazam + MusicBrainz recognition with optional random sampling.

Combines the original shazam_year.py with the sampling logic from shazample.sh.

Requires one of: songrec, ShazamAPI, or shazamio for recognition.
If songrec fails, it will fall back to ShazamAPI.
Asyncio recognition via shazamio can be attempted first with the --asyncio-first flag.
"""

import argparse
import asyncio
import logging
import math
import random
import socket
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

import dateutil.parser
import musicbrainzngs
import pprintpp
import regex as re
import requests
import simplejson
from path import Path
import sh
from tqdm import tqdm

# ----------------------------------------------------------------------
# Constants & setup
# ----------------------------------------------------------------------
musicbrainzngs.set_useragent("shazam_year_unified.py", "1.0.0", "andy80586@gmail.com")

DEFAULT_RESULT_LIMIT = 400
DEFAULT_MATCH_COUNT = 3
DEFAULT_SHAZAM_INTERVAL = 8.0
DEFAULT_TIMEOUT = 10.0
DEFAULT_NUM_SAMPLES = 1
DEFAULT_SAMPLE_DURATION = 30
SAMPLE_METHODS = ("uniform", "biased")

LOG = logging.getLogger()


class MusicBrainzTimeout(TimeoutError):
    """Raised when a MusicBrainz operation exceeds its timeout."""

    pass


@contextmanager
def musicbrainz_timeout(seconds: float = 10) -> Generator[None, Any, None]:
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    except (socket.timeout, TimeoutError) as exc:
        raise MusicBrainzTimeout(
            f"MusicBrainz request timed out after {seconds}s"
        ) from exc
    finally:
        socket.setdefaulttimeout(previous_timeout)


# ----------------------------------------------------------------------
# Existing helper functions (unchanged)
# ----------------------------------------------------------------------
def _parse_year_from_shazam_data(shazam_data):
    """Return Shazam's release year, or None when it is absent."""
    try:
        song_metadata = shazam_data["track"]["sections"][0]["metadata"]
        for mdict in song_metadata:
            if mdict.get("title") == "Released":
                return mdict.get("text")
        return None
    except (KeyError, IndexError, TypeError):
        return None


def strip_parens(text, delete_text_inside=True):
    """Remove round/square brackets, optionally including their contents."""
    stripped_text = text
    for grouping in ["()", "[]"]:
        if delete_text_inside:
            sub_rgx = "(\\s*[{g[0]}][^{g[1]}]+[{g[1]}])".format(g=grouping)
            stripped_text = re.sub(sub_rgx, "", stripped_text)
        else:
            stripped_text = stripped_text.replace(grouping[0], "").replace(grouping[1], "")
    return stripped_text


def fetch_local_shazam_data(
        song_path,
        match_count=DEFAULT_MATCH_COUNT,
        max_time_seconds=DEFAULT_SHAZAM_INTERVAL,
):
    """Recognize a song through the local ShazamAPI fallback."""
    from ShazamAPI import Shazam

    content = Path(song_path).bytes()
    shazam = Shazam(content)
    shazam.MAX_TIME_SECONDS = int(max_time_seconds)
    recognizer = shazam.recognizeSong()
    song_data_by_artist_title = {}
    total_length = float(sh.soxi("-D", song_path, _err=None))
    with tqdm(total=total_length, desc="Querying Shazam...") as pbar:
        for offset, song_data in recognizer:
            pbar.update(offset - pbar.n)
            if not song_data.get("matches"):
                continue
            if not song_data.get("track"):
                continue
            artist, title = extract_artist_title_from_track(song_data["track"])
            print(f"Match found {{artist: {artist}, title: {title}, offset: {offset}}}")
            song_data_by_artist_title.setdefault((artist, title), []).append(song_data)
            if len(song_data_by_artist_title[(artist, title)]) >= match_count:
                break
    sorted_song_data = sorted(
        song_data_by_artist_title.items(),
        key=lambda kv: len(kv[1]),
        reverse=True
    )
    for (_, title), song_data_list in sorted_song_data:
        if _parse_year_from_shazam_data(song_data_list[0]):
            return song_data_list[0]
    if sorted_song_data:
        return sorted_song_data[0][1][0]
    raise Exception("No song data came back from ShazamAPI!")


def fetch_shazam_data(song_path, match_count=DEFAULT_MATCH_COUNT, max_time_seconds=DEFAULT_SHAZAM_INTERVAL):
    """Recognize a song through the songrec command-line client."""
    try:
        songrec_output = sh.songrec.recognize("-j", song_path, _timeout=DEFAULT_TIMEOUT)
    except sh.ErrorReturnCode as exc:
        raise Exception("No song data came back from songrec!") from exc
    return simplejson.loads(songrec_output)


def _count_recordings(artist, title):
    with musicbrainz_timeout(seconds=DEFAULT_TIMEOUT):
        return musicbrainzngs.search_recordings(
            artist=artist,
            recording=title,
            strict=True
        )["recording-count"]


def _escape_lucene_phrase(value):
    """Escape characters that are significant inside a quoted Lucene phrase."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


_APOSTROPHE_RE = re.compile(r"['\u2018\u2019\u201b\u02bc\uff07]")


def _normalize_lucene_title_phrase(value):
    """Make punctuation variants produce the same MusicBrainz search terms."""
    value = _APOSTROPHE_RE.sub("", value)
    value = re.sub(r"\p{P}", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _strip_leading_the(value):
    return re.sub(r"^the\s+", "", value, flags=re.IGNORECASE).strip()


def _build_artist_query(artist):
    artists = [
        _strip_leading_the(part)
        for part in re.split(
            r"\s+(?:and|&|pres(?:ents)?|feat(?:uring)?\.?|ft\.?)\s+",
            artist,
            flags=re.IGNORECASE,
        )
    ]
    queries = [
        f'artist:"{_escape_lucene_phrase(name)}"'
        for name in artists
        if name
    ]
    if len(queries) == 1:
        return queries[0]
    return f"({' OR '.join(queries)})"


def _build_title_query(title):
    alternative_queries = []
    for alternative in re.split(r"\s*/\s*", title):
        titles = [
            _strip_leading_the(_normalize_lucene_title_phrase(part))
            for part in re.split(
                r"\s+(?:and|&)\s+",
                alternative,
                flags=re.IGNORECASE,
            )
        ]
        queries = [
            f'recording:"{_escape_lucene_phrase(name)}"'
            for name in titles
            if name
        ]
        if len(queries) == 1:
            alternative_queries.append(queries[0])
        elif queries:
            alternative_queries.append(f"({' AND '.join(queries)})")

    if len(alternative_queries) == 1:
        return alternative_queries[0]
    return f"({' OR '.join(alternative_queries)})"


def _query_recordings(artist, title, first_only=False, result_limit=DEFAULT_RESULT_LIMIT):
    normalized_artist = normalize_artist(artist)
    normalized_title = normalize_title(title)
    query = (
        f"{_build_artist_query(normalized_artist)} AND "
        f"{_build_title_query(normalized_title)}"
    )
    LOG.info("Strict MusicBrainz query: %s", query)

    all_recordings = []
    search_args_list = [
        {"query": query, "strict": True},
        {
            "artist": normalized_artist,
            "recording": normalized_title,
            "strict": False,
        },
    ]
    if first_only:
        search_args_list = search_args_list[:1]

    for search_args in search_args_list:
        offset = 0
        while offset < result_limit:
            LOG.info(
                "Fetching MusicBrainz recordings %s-%s...",
                offset + 1,
                offset + 100,
            )
            with musicbrainz_timeout(seconds=DEFAULT_TIMEOUT):
                recording_result = musicbrainzngs.search_recordings(
                    offset=offset,
                    limit=100,
                    **search_args,
                )
            recordings = recording_result["recording-list"]
            if not recordings:
                break

            all_recordings.extend(recordings)

            if len(recordings) < 100:
                break
            offset += 100

    releases_for_recordings = []
    for rec in all_recordings:
        for release in rec.get("release-list", []):
            release['artist-credit'] = release.setdefault('artist-credit', []) + (rec['artist-credit'] or [])
        releases_for_recordings.extend(rec.get("release-list", []))

    for release in releases_for_recordings:
        if not release.get("date"):
            release["date"] = datetime.now().strftime("%Y-%m-%d")
    if not releases_for_recordings:
        return {
            "releases": [],
            "reldate": datetime(2099, 12, 31),
            "recdate": datetime(2099, 12, 31)
        }

    matching_releases = []
    artist_to_match = re.sub(
        "\\p{Pd}", " ", re.sub(
            "(?i)\\s+(?:&|and|pres(?:ents)?)\\s+", " and ", artist
        )
    )
    artist_to_match = "|".join(
        set(
            [artist_to_match, artist] + re.split(
                "(?i)\\s+and\\s+", artist_to_match
            )
        )
    )
    artist_to_match = artist_to_match.replace("philharmonic", "(?:philharmonic|symphony|festival)")
    artist_to_match = artist_to_match.replace("?", "\\?")

    title_to_match = re.sub(
        "\\p{Pd}",
        " ",
        re.sub(
            "(?i)\\s+(?:&|and)\\s+",
            " and ",
            title,
        )
    )
    title_to_match = "|".join(
        set(
            [title_to_match, title] + re.split(
                "\\s*/\\s*", title_to_match
            )
        )
    )

    for release in releases_for_recordings:
        artists = [re.sub("(?i)\\s+(?:&|and|pres(?:ents)?)\\s+", " and ", ac.get("name", "")) for ac in
                   release.get("artist-credit", [{}]) if isinstance(ac, dict)]
        if any(re.search(artist_to_match, artist, re.IGNORECASE) for artist in artists):
            titles = [tl.get("title", "") for ml in release.get("medium-list", [{}]) for tl in
                      ml.get("track-list", [{}])]
            titles = [re.sub("\\p{P}", "", title) for title in titles]
            title_to_match_without_punctuation = re.sub("\\p{P}", "", title_to_match)
            if any(re.match(title_to_match_without_punctuation, title, re.IGNORECASE) for title in titles):
                matching_releases.append(release)

    matching_recordings = []
    for recording in all_recordings:
        release_matched = False
        for release in matching_releases:
            if release in recording.get("release-list", []):
                release_matched = True
                break
        if release_matched:
            matching_recordings.extend(recording.get("release-list", []))

    return {
        "releases": matching_releases,
        "recordings": matching_recordings,
        "reldate": min((dateutil.parser.parse(r["date"]) for r in
                        matching_releases)) if matching_releases else dateutil.parser.parse("2099"),
        "recdate": min((dateutil.parser.parse(r["date"]) for r in
                        matching_recordings)) if matching_recordings else dateutil.parser.parse("2099"),
    }


_MIXED_TAG_RE = re.compile(r"\s*\{MIXED}\s*", re.IGNORECASE)
_TRAILING_Q_RE = re.compile(r"\?+\s*$")
_LEADING_THE_RE = re.compile(r"(?i)^\s*the\s+")
_BAD_SEP_RE = re.compile(r"\s*[/\\]\s*")
_MULTI_SPACE_RE = re.compile(r"\s+")

_STRIP_QUOTES_RE = re.compile(
    r"[\u0022\u0027\u00ab\u00bb\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f\u2039\u203a"
    r"\u2e42\u300c\u300d\u300e\u300f\u301d\u301e\u301f"
    r"\ufe41\ufe42\ufe43\ufe44\uff02\uff07\uff62\uff63]+"
)
_DOUBLE_ASTERISK_BLOCK_RE = re.compile(r"\*{2,}.*?\*{2,}", re.IGNORECASE)


def normalize_artist(artist: str) -> str:
    """Normalize noisy Shazam artist text without changing its meaning."""
    if not artist:
        return artist
    artist = artist.strip()
    artist = _MIXED_TAG_RE.sub(" ", artist)
    artist = _TRAILING_Q_RE.sub("", artist)
    artist = _MULTI_SPACE_RE.sub(" ", artist)
    return artist.strip()


def normalize_title(title: str) -> str:
    """Normalize title decorations that should not reach MusicBrainz."""
    if not title:
        return title

    # Remove **...** blocks (e.g. Massive **tune Of The Week**)
    title = _DOUBLE_ASTERISK_BLOCK_RE.sub(" ", title)

    title = strip_parens(title).strip()
    title = _MIXED_TAG_RE.sub(" ", title)
    title = _TRAILING_Q_RE.sub("", title)

    # If stray asterisks remain, remove them
    title = re.sub(r"\*+", "", title)

    title = _MULTI_SPACE_RE.sub(" ", title)
    return title.strip()


def sanitize_filename_component(text: str) -> str:
    """Remove characters that make artist/title filename components unsafe."""
    if not text:
        return text
    text = _STRIP_QUOTES_RE.sub("", text)
    text = _BAD_SEP_RE.sub(" - ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def build_target_filename(artist: str, title: str, year: int, ext: str = ".mp3") -> str:
    """Build the canonical filename for a recognized song."""
    artist_n = sanitize_filename_component(normalize_artist(artist))
    title_n = sanitize_filename_component(normalize_title(title))
    filename = f"{artist_n} - {title_n} ({year}){ext}"
    filename = _LEADING_THE_RE.sub("", filename)
    filename = _MULTI_SPACE_RE.sub(" ", filename)
    return filename.strip()


def extract_artist_title_from_track(track: dict) -> tuple[str, str]:
    """Return a normalized artist/title pair from a Shazam track."""
    artist = track.get("subtitle")
    title = track.get("title")
    return normalize_artist(artist), normalize_title(title)


def extract_artist_title_from_filename(song_path) -> tuple[str, str]:
    """Extract a normalized artist/title pair from a canonical song filename."""
    filename = Path(song_path).basename()
    stem = Path(filename).splitext()[0]
    match = re.fullmatch(
        r"(?P<artist>.+?)\s+-\s+(?P<title>.+?)(?:\s+\(\d{4}\))?",
        stem,
    )
    if match is None:
        raise ValueError(
            f"Cannot extract artist and title from filename: {filename}"
        )
    return normalize_artist(match["artist"]), normalize_title(match["title"])


def fetch_similar_tracks(shazam_data):
    """Return normalized similar tracks for any successful Shazam response."""
    track = shazam_data["track"]
    default_url = (
        "https://cdn.shazam.com/shazam/v3/en-US/GB/iphone/-/tracks/"
        f"track-similarities-id-{track['key']}"
        "?startFrom=0&pageSize=20&connected="
    )
    url = track.get("relatedtracksurl", default_url)
    response = requests.get(url, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return [
        {
            "artist": normalize_artist(item["subtitle"]),
            "title": normalize_title(item["title"]),
        }
        for item in response.json()["tracks"]
    ]


def recognize_audio(song_path, asyncio_first=False, match_count=DEFAULT_MATCH_COUNT,
                    query_interval=DEFAULT_SHAZAM_INTERVAL, local_fallback=True):
    """Recognize one audio file using the configured fallback chain."""
    if asyncio_first:
        try:
            from shazamio import Shazam as ShazamAlt
            result = asyncio.run(ShazamAlt(endpoint_country="US").recognize(song_path))
            if result.get("track"):
                return result
        except Exception:
            LOG.debug("Async recognition failed", exc_info=True)
    try:
        return fetch_shazam_data(
            song_path, match_count=match_count, max_time_seconds=query_interval
        )
    except Exception:
        if not local_fallback:
            raise
        LOG.debug("songrec recognition failed; trying ShazamAPI", exc_info=True)
        return fetch_local_shazam_data(
            song_path, match_count=match_count, max_time_seconds=query_interval
        )


# ----------------------------------------------------------------------
# Shazam query wrapper (updated to use normalization helpers)
# ----------------------------------------------------------------------
def query_shazam(
        song_path,
        first_only=False,
        match_count=DEFAULT_MATCH_COUNT,
        result_limit=DEFAULT_RESULT_LIMIT,
        query_interval=DEFAULT_SHAZAM_INTERVAL,
        show_similar=False,
        asyncio_first=False,
        year_only=False,
):
    shazam_data = {}
    artist = None
    title = None
    if not year_only:
        shazam_data = recognize_audio(
            song_path, asyncio_first=asyncio_first, match_count=match_count,
            query_interval=query_interval,
        )
        artist, title = extract_artist_title_from_track(shazam_data["track"])
    else:
        artist, title = extract_artist_title_from_filename(song_path)

    parsed_year = int(_parse_year_from_shazam_data(shazam_data) or "2099")
    if not show_similar:
        year_data = _query_recordings(
            artist=artist,
            title=title,
            first_only=first_only,
            result_limit=result_limit
        )
        possible_years = [year_data["reldate"].year, year_data["recdate"].year, parsed_year]
        if all((
                year_data["reldate"].year == year_data["recdate"].year,
                parsed_year < year_data["reldate"].year,
                year_data["reldate"].year != 2099,
        )):
            LOG.warning(
                "Shazam returned a release year of %s, but querying the releases and recordings on "
                "MusicBrainz yielded a later year of %s! Double-check to ensure accuracy.",
                parsed_year, year_data["reldate"].year
            )
    else:
        possible_years = [parsed_year]
    result = {
        "artist": artist,
        "title": title,
        "year": min(possible_years),
        "similar": [],
        "raw": shazam_data
    }
    if show_similar:
        try:
            result["similar"] = fetch_similar_tracks(shazam_data)
        except Exception:
            LOG.error("Error getting related tracks", exc_info=True)
    return result


# ----------------------------------------------------------------------
# New sampling functions
# ----------------------------------------------------------------------
def get_song_duration(song_path):
    """Return duration in seconds using soxi."""
    return float(sh.soxi("-D", song_path, _err=None))


def generate_sample_start(method, song_duration, sample_duration):
    """Generate a random start time for a sample."""
    max_start = song_duration - sample_duration
    if max_start <= 0:
        return 0.0

    if method == "uniform":
        return random.uniform(0, max_start)
    if method == "biased":
        # Biased normal distribution around midpoint (replicates shazample.sh)
        midpoint = song_duration / 2.0
        # Box-Muller transform (as in the awk script)
        u1 = max(random.random(), sys.float_info.min)
        u2 = random.random()
        z = (-2 * math.log(u1)) ** 0.5 * math.cos(2 * math.pi * u2)
        scale = 0.3
        offset = z * scale  # mean 0, stddev ~0.3
        start = midpoint + offset * midpoint
        # Clamp to valid range
        return max(0.0, min(start, max_start))
    raise ValueError(f"Unknown sampling method: {method}")


def extract_sample(song_path, start, duration, suffix=".wav"):
    """Extract a segment from song_path into a temporary file. Return the temp file path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp_path = temp.name
    try:
        sh.sox(song_path, temp_path, "trim", str(start), str(duration), _out="/dev/null", _err="/dev/null")
    except Exception:
        Path(temp_path).unlink_p()
        raise
    return temp_path


def recognize_sample(sample_path, asyncio_first=False, match_count=DEFAULT_MATCH_COUNT,
                     query_interval=DEFAULT_SHAZAM_INTERVAL):
    """
    Recognise a single sample file.
    Returns a dict with artist, title, raw_data or None on failure.
    """
    try:
        data = recognize_audio(
            sample_path, asyncio_first=asyncio_first, match_count=match_count,
            query_interval=query_interval, local_fallback=False,
        )
        artist, title = extract_artist_title_from_track(data["track"])
        year = int(_parse_year_from_shazam_data(data) or "2099")
        return {"artist": artist, "title": title, "year": year, "raw": data}
    except Exception as exc:
        LOG.debug("Recognition failed for sample: %s", exc)
        return None


def recognize_song_samples(
        song_path, num_samples=DEFAULT_NUM_SAMPLES,
        sample_duration=DEFAULT_SAMPLE_DURATION, sample_method='uniform',
        consensus=False, asyncio_first=False, match_count=DEFAULT_MATCH_COUNT,
        query_interval=DEFAULT_SHAZAM_INTERVAL,
):
    """Recognize sampled segments and return the winning Shazam result."""
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if sample_method not in SAMPLE_METHODS:
        raise ValueError(f"Unknown sampling method: {sample_method}")

    song_path = Path(song_path)
    song_duration = get_song_duration(song_path)
    if song_duration < sample_duration:
        raise ValueError(
            f"Track is too short for sampling (duration {song_duration:.2f} "
            f"< sample duration {sample_duration:.2f})"
        )

    LOG.info("Sampling %d random segments of %.2f seconds each (method: %s)",
             num_samples, sample_duration, sample_method)
    sample_results = []
    pair_counter = Counter()
    majority_needed = (num_samples // 2) + 1

    for sample_number in range(1, num_samples + 1):
        start = generate_sample_start(sample_method, song_duration, sample_duration)
        LOG.debug("Sample %d start: %.2f", sample_number, start)
        sample_path = extract_sample(song_path, start, sample_duration)
        try:
            result = recognize_sample(
                sample_path, asyncio_first=asyncio_first,
                match_count=match_count, query_interval=query_interval,
            )
            if not result:
                LOG.warning("→ Sample %d at %.2f sec: no match", sample_number, start)
                continue
            sample_results.append(result)
            pair = (result['artist'], result['title'])
            pair_counter[pair] += 1
            LOG.info("→ Sample %d at %.2f sec: %s - %s",
                     sample_number, start, result['artist'], result['title'])
            if consensus and pair_counter[pair] >= majority_needed:
                LOG.info("Consensus reached for (%s - %s) after %d samples.",
                         result['artist'], result['title'], sample_number)
                break
            if not consensus:
                break
        finally:
            Path(sample_path).unlink_p()

    if not sample_results:
        raise RuntimeError(f"No sampled segment of {song_path.basename()} was recognized")
    if not consensus:
        return sample_results[0]

    winning_pair, count = pair_counter.most_common(1)[0]
    LOG.info("Consensus artist/title: %s - %s (appeared %d/%d times)",
             winning_pair[0], winning_pair[1], count, len(sample_results))
    winning_results = [result for result in sample_results
                       if (result['artist'], result['title']) == winning_pair]
    winner = dict(winning_results[0])
    winner['year'] = min(result['year'] for result in winning_results)
    return winner


def recognize_song(song_path, options):
    """Run either recognition strategy and return one common result shape."""
    if options.num_samples == 1 or options.year_only:
        return query_shazam(
            song_path,
            match_count=options.match_count,
            first_only=options.first_only,
            result_limit=options.limit,
            query_interval=options.interval,
            show_similar=options.similar,
            asyncio_first=options.asyncio_first,
            year_only=options.year_only,
        )

    result = recognize_song_samples(
        song_path,
        num_samples=options.num_samples,
        sample_duration=options.sample_duration,
        sample_method=options.sample_method,
        consensus=options.consensus,
        asyncio_first=options.asyncio_first,
        match_count=options.match_count,
        query_interval=options.interval,
    )
    year_data = _query_recordings(
        artist=result["artist"],
        title=result["title"],
        first_only=options.first_only,
        result_limit=options.limit,
    )
    result["year"] = min(
        result["year"], year_data["reldate"].year, year_data["recdate"].year
    )
    result["similar"] = []
    if options.similar:
        try:
            result["similar"] = fetch_similar_tracks(result["raw"])
        except Exception:
            LOG.error("Error getting related tracks", exc_info=True)
    return result


def rename_song(song_path, result):
    """Rename a recognized file without overwriting an existing target."""
    extension = song_path.splitext()[1] or ".mp3"
    filename = build_target_filename(
        result["artist"], result["title"], result["year"], ext=extension
    )
    destination = song_path.dirname().joinpath(filename)
    if destination == song_path:
        LOG.info("%s: file name is already correct", song_path.basename())
    elif destination.exists():
        LOG.error("%s: target file already exists; not renaming", destination)
    else:
        LOG.info("%s: renaming to '%s'", song_path.basename(), filename)
        song_path.rename(destination)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def build_argument_parser():
    """Create the command-line parser."""
    ap = argparse.ArgumentParser(
        description="Recognise songs using Shazam and lookup year on MusicBrainz. "
                    "Optionally sample random segments from a song."
    )
    ap.add_argument("-v", "--verbose", action="store_true", default=False,
                    help="Enable verbose mode - print all the song data")
    ap.add_argument("-r", "--rename", action="store_true", default=False,
                    help="Rename file(s) to <artist> - <title> (year).mp3")
    ap.add_argument("-f", "--first-only", action="store_true", default=False,
                    help="First query set only")
    ap.add_argument("-a", "--asyncio-first", action="store_true", default=False,
                    help="Use asyncio recognizer first")
    ap.add_argument("-m", "--match-count", type=int, default=DEFAULT_MATCH_COUNT,
                    help="Match the artist and title N times before breaking (default: %(default)s)")
    ap.add_argument("-s", "--similar", action="store_true", default=False,
                    help="List the top 20 similar artists and titles")
    ap.add_argument("-l", "--limit", type=int, default=DEFAULT_RESULT_LIMIT,
                    help="Limit searches to N (default: %(default)s) results")
    ap.add_argument("-i", "--interval", type=float, default=DEFAULT_SHAZAM_INTERVAL,
                    help="The interval of music to query Shazam with (default: %(default)s sec)")
    ap.add_argument("-y", "--year-only", action="store_true", default=False,
                    help="Only query the year of the song")

    ap.add_argument("-n", "--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                    help="Number of random samples to take (default: 1 = no sampling)")
    ap.add_argument("-d", "--sample-duration", type=float, default=DEFAULT_SAMPLE_DURATION,
                    help="Duration of each sample in seconds (default: %(default)s)")
    ap.add_argument("--sample-method", choices=SAMPLE_METHODS, default="uniform",
                    help="Random sampling method: uniform or biased (default: uniform)")
    ap.add_argument("--consensus", action="store_true", default=False,
                    help="When sampling, use the most frequent artist/title from all samples "
                         "for year lookup and renaming (otherwise use first successful sample)")

    ap.add_argument("song_paths", type=Path, nargs="+", help="One or more song paths to query")
    return ap


def main(argv=None):
    """Run the command-line interface."""
    ns = build_argument_parser().parse_args(argv)

    # Logging setup
    if ns.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger("musicbrainzngs").setLevel(logging.ERROR)

    for song_path in ns.song_paths:
        LOG.info("Processing: %s", song_path.basename())
        try:
            result = recognize_song(song_path, ns)
        except Exception:
            LOG.exception("%s: recognition failed", song_path.basename())
            continue

        LOG.info(
            "Recognized: %s - %s (%s)",
            result["artist"], result["title"], result["year"],
        )
        if ns.rename:
            rename_song(song_path, result)
        if result["similar"]:
            LOG.info(
                "Similar music:\n%s",
                "\n".join(
                    f"{track['artist']} - {track['title']}"
                    for track in result["similar"]
                ),
            )
        if ns.verbose and result["raw"]:
            LOG.info("Raw Shazam result:\n%s", pprintpp.pformat(result["raw"]))


if __name__ == "__main__":
    main()
