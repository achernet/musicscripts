#!/usr/bin/env python3
"""Find and optionally download songs related to a recognized local track."""

import argparse
import ast
import json
import logging
import re
import sys
import tempfile
from typing import Dict, List, Optional

import requests
import sh
from lxml import etree
from lxml.html import HTMLParser
from path import Path

from shazam_year import recognize_song_samples, sanitize_filename_component
from youtube_scorer import fetch_youtube_results, score_youtube_results

LOG = logging.getLogger(__name__)

SHAZAM_NUM_SAMPLES = 11
SHAZAM_SAMPLE_DURATION = 12
HTTP_TIMEOUT = 10
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 10; K) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Mobile Safari/537.36"
    ),
    "Sec-CH-UA": (
        '"Not=A?Brand";v="99", '
        '"Google Chrome";v="151", '
        '"Chromium";v="151"'
    ),
}


def extract_json_from_element(element):
    """Extract the first JSON object embedded in a JavaScript string."""
    try:
        matcher = re.finditer("[^\\\\]\"", element.text)
        first = next(matcher)
        last = next(matcher)
    except (StopIteration, TypeError) as exc:
        raise ValueError("Element does not contain a quoted JSON string") from exc
    start = first.end() - 1
    end = last.end()
    unescaped_string = ast.literal_eval(element.text[start:end])

    object_start = re.search(r"[\[{]", unescaped_string)
    if not object_start:
        raise ValueError("No JSON object found in element text")
    end_char = "]" if object_start.group() == "[" else "}"
    object_end = unescaped_string.rfind(end_char)
    if object_end < object_start.start():
        raise ValueError("JSON object is missing its closing delimiter")
    object_string = unescaped_string[object_start.end() - 1:object_end + 1]
    return json.loads(object_string)


def can_extract_json(element):
    """Return whether an element contains parseable embedded JSON."""
    try:
        extract_json_from_element(element)
        return True
    except Exception:
        return False


def get_shazam_url(file_path):
    """Recognize a file and return its canonical Shazam track URL."""
    shazam_data = recognize_song_samples(
        file_path,
        num_samples=SHAZAM_NUM_SAMPLES,
        sample_duration=SHAZAM_SAMPLE_DURATION,
        consensus=True,
    )["raw"]
    try:
        url = shazam_data["track"]["url"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Shazam did not return a track URL for {file_path}") from exc
    if not url:
        raise RuntimeError(f"Shazam did not return a track URL for {file_path}")

    LOG.info("Shazam resolved track %s to URL: %s", Path(file_path).basename(), url)
    return url


def normalize_query(text):
    """Remove leading articles and bracketed version annotations."""
    # 1. Remove leading "The"
    text = re.sub(r'^The\s+', '', text, flags=re.IGNORECASE)

    # 2. Repeatedly remove innermost brackets (handles nesting)
    patterns = [r'\([^()]*\)', r'\[[^[\]]*\]', r'\{[^{}]*\}']
    for pattern in patterns:
        while re.search(pattern, text):
            text = re.sub(pattern, '', text)

    return text.strip()


def parse_song_parts(
    artist_part: str,
    title: str,
    file: Optional[Path] = None,
) -> Dict[str, Optional[object]]:
    """Normalize separate artist and title fields into song data."""
    title = normalize_query(title)

    # Normalize artist separators
    # Replace & with comma for consistent splitting
    artist_part = artist_part.replace("&", ",")

    # Split by comma
    raw_artists = [a.strip() for a in artist_part.split(",")]

    artists: List[str] = []
    for artist in raw_artists:
        artist = normalize_query(artist)
        if artist and artist not in artists:
            artists.append(artist)

    return {
        "artists": artists,
        "title": title,
        "file": file,
    }


def parse_song_filename(song: str) -> Dict[str, Optional[object]]:
    """Parse an artist-title MP3 filename into normalized song data."""
    # Remove directory info if present
    song = Path(song).basename()

    # Store the base name of the file in the result
    file = Path(song)

    # Remove extension
    song = re.sub(r'\.mp3$', '', song, flags=re.IGNORECASE).strip()

    # Extract year from final (YYYY)
    year_match = re.search(r'\((\d{4})\)\s*$', song)
    if year_match:
        song = song[:year_match.start()].strip()

    # Remove bracket junk like [Aly & Fila vs...]
    song = re.sub(r'\[.*?]', '', song).strip()

    # Split artist and title on first hyphen
    if " - " not in song:
        raise ValueError(f"Invalid format: missing ' - ' separator in song: {song}")

    artist_part, title = song.split(" - ", 1)

    return parse_song_parts(artist_part, title, file=file)


def invoke_shazam(file_path, files_by_title=None):
    """Fetch and parse the related-song metadata from a Shazam track page."""
    url = get_shazam_url(file_path)
    shazam_page = requests.get(
        url,
        headers=HTTP_HEADERS,
        timeout=HTTP_TIMEOUT,
    )
    try:
        shazam_page.raise_for_status()
    except requests.HTTPError:
        LOG.critical("Error loading Shazam url: %s", url, exc_info=True)
        raise
    tree = etree.fromstring(shazam_page.text, HTMLParser())
    tree.make_links_absolute(shazam_page.url)
    linked_song_elements = tree.xpath('body/script[contains(text(), \'inAlbum\')]')
    if not linked_song_elements:
        raise RuntimeError("Shazam page did not contain related-song metadata")
    linked_songs_elem = linked_song_elements[0]
    linked_song_data = extract_json_from_element(linked_songs_elem)
    if linked_song_data['url'] != shazam_page.url:
        LOG.critical("Shazam URL mismatch: %s != %s", linked_song_data['url'], shazam_page.url)
        tempdir = Path(tempfile.mkdtemp())
        with (tempdir / "shazam.json").open("w") as json_file:
            json.dump(linked_song_data, json_file, indent=2)
        with (tempdir / "shazam.html").open("w") as html_file:
            html_file.write(
                etree.tostring(
                    tree,
                    method="html",
                    encoding="unicode",
                    pretty_print=True,
                )
            )
        LOG.info("Saved mismatched Shazam response to %s", tempdir)

        raise RuntimeError("Shazam URL mismatch")
    linked_song_dicts = linked_song_data['citation']

    song_data_list = []
    for song in linked_song_dicts:
        # Shazam already provides separate fields. Do not turn them into a
        # filename: legitimate titles can contain path separators such as
        # "The Letter (Live At The Fillmore East/1970)".
        song_data = parse_song_parts(song["byArtist"], song["name"])
        if "inAlbum" in song:
            song_data["match"] = "artist"
        elif "byArtist" in song:
            song_data["match"] = "similar"
        else:
            song_data["match"] = "unknown"
        song_data["file"] = already_downloaded_path(song_data, files_by_title)
        if song_data not in song_data_list:
            song_data_list.append(song_data)

    return song_data_list


def _build_downloaded_index() -> Dict[str, List[Dict[str, Optional[object]]]]:
    """Index downloaded songs by normalized lowercase title."""
    files_by_title: Dict[str, List[Dict[str, Optional[object]]]] = {}
    filenames = Path("~/Music/random").expanduser().files("*.mp3")
    for filename in filenames:
        try:
            filename_object = parse_song_filename(filename)
            next_title_key = filename_object["title"].lower()
            files_by_title.setdefault(next_title_key, []).append(filename_object)
        except ValueError:
            continue
    return files_by_title


def already_downloaded_path(
    song_data: Dict[str, Optional[object]],
    files_by_title: Optional[Dict[str, List[Dict[str, Optional[object]]]]] = None,
) -> Optional[str]:
    """
    If anything in ~/Music/random has a matching title and at least one of the same artists,
    we've already downloaded this song data.
    """
    if files_by_title is None:
        files_by_title = _build_downloaded_index()

    title_key = sanitize_filename_component(song_data["title"]).lower()

    # If any of the artists match, return its path.
    artist_set = {sanitize_filename_component(artist).lower() for artist in song_data['artists']}
    for file_object in files_by_title.get(title_key, []):
        file_artists = {sanitize_filename_component(artist).lower() for artist in file_object['artists']}
        if artist_set & file_artists:
            return file_object['file']

    # None of the artists matched, or we got an empty list
    return None


def run_command_showing_output(command: sh.Command):
    """Run a command while forwarding its text or byte output immediately."""
    def print_live(chunk):
        if isinstance(chunk, bytes):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        else:
            sys.stdout.write(chunk)
            sys.stdout.flush()

    result = command(_out=print_live, _err=sys.stderr, _out_bufsize=0, _return_cmd=True)
    return result


def fetch_song(youtube_url: str):
    """Download, normalize, recognize, and rename one YouTube result."""
    yt_dlp = sh.Command("yt-dlp")
    command = yt_dlp.bake(
        "-R", 3,
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "192k",
        "-P", Path("~/Music/random").expanduser(),
        "--exec",
        f"mp3gain -r -k -p {{}} && "
        f"python3 shazam_year.py -r -n {SHAZAM_NUM_SAMPLES} -d {SHAZAM_SAMPLE_DURATION} --consensus {{}}",
    )
    command_result = run_command_showing_output(command.bake(youtube_url))
    return command_result


def youtube_result_url(result: dict) -> str:
    """Return the canonical URL for a yt-dlp search result."""
    url = result.get("webpage_url") or result.get("url")
    if url and str(url).startswith(("http://", "https://")):
        return str(url)
    video_id = result.get("id") or url
    if not video_id:
        raise ValueError("YouTube result does not contain a URL or video ID")
    return f"https://www.youtube.com/watch?v={video_id}"


def total_match_weighted_views(results: List[dict]) -> int:
    """Estimate total relevant views across results matching artist and title."""
    total = 0.0
    for result in results:
        try:
            views = int(result.get("view_count") or 0)
            artist_score = float(result.get("artist_match_score", 0.0))
            title_score = float(result.get("title_boost", 0.0))
        except (TypeError, ValueError):
            continue

        # Both the artist and title must match. Their product discounts results
        # where either half of the query is weak while retaining fuzzy matches.
        joint_match_score = artist_score * title_score
        total += views * joint_match_score
    return round(total)


def display_selected_youtube_result(
    song_data: dict,
    result: dict,
    all_results: Optional[List[dict]] = None,
) -> None:
    """Show the query and selected video metadata side by side."""
    artists = ", ".join(song_data["artists"])
    channel = result.get("uploader") or result.get("channel") or "?"
    title = result.get("title") or "?"
    views = result.get("view_count")
    try:
        views_text = f"{int(views):,}"
    except (TypeError, ValueError):
        views_text = "?"
    duration = result.get("duration_string")
    if not duration:
        duration_seconds = result.get("duration")
        duration = f"{duration_seconds} sec" if duration_seconds is not None else "?"

    artist_score = float(result.get("artist_match_score", 0.0))
    title_score = float(result.get("title_boost", 0.0))
    mismatch = artist_score < 0.5 or title_score < 0.5
    status = "LIKELY MISMATCH" if mismatch else "MATCH"

    print(f"Query:   {artists} - {song_data['title']}")
    print(f"Result:  {channel} - {title}")
    if all_results is not None:
        matched_views = total_match_weighted_views(all_results)
        print(f"Total matched views: {matched_views:,}")
    print(f"Selected URL views: {views_text}")
    print(f"Duration: {duration}")
    print(
        f"Match:   {status} "
        f"(artist: {artist_score:.0%}, title: {title_score:.0%})"
    )
    print(f"URL:     {youtube_result_url(result)}")


def fetch_similar_songs(song_path: str, list_only: bool = False):
    """Find, rank, display, and optionally download Shazam-related songs."""
    files_by_title = _build_downloaded_index()
    song_data_list = invoke_shazam(song_path, files_by_title)
    for song_data in song_data_list:
        file_path = song_data.get("file")
        if file_path and not list_only:
            continue

        results = fetch_youtube_results(
            track_name=song_data['title'],
            artists=song_data['artists'],
            quantity=30,
        )
        scored = score_youtube_results(results, song_data['title'], song_data['artists'])
        if not scored:
            LOG.warning(
                "No results came back for title: %(title)s or for artists: %(artists)s, skipping...",
                song_data,
            )
            continue

        selected = scored[0]
        youtube_url = youtube_result_url(selected)
        display_selected_youtube_result(song_data, selected, scored)
        print()
        if not list_only:
            fetch_song(youtube_url)
    return song_data_list


def build_argument_parser():
    """Create the command-line parser."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d",
        "--download",
        action="store_true",
        help="Download similar songs and their best YouTube URLs",
    )
    ap.add_argument("song_path", help="The song path")
    return ap


def main(argv=None):
    """Run the command-line interface."""
    ns = build_argument_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    song_data_list = fetch_similar_songs(ns.song_path, list_only=not ns.download)
    if ns.download:
        print(song_data_list)


if __name__ == "__main__":
    main()
