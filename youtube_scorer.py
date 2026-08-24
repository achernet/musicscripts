#!/usr/bin/env python3
"""Search YouTube for a recording and rank candidates by match quality."""

import argparse
import math
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, cast

import numpy as np
import simplejson
import yt_dlp
from rapidfuzz.fuzz import partial_ratio, token_sort_ratio
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text
from scipy.signal import find_peaks
from unidecode import unidecode
from yt_dlp.utils import DownloadError

Cluster = Tuple[float, float, int]  # (center, sigma, count)
KeywordMatch = Tuple[str, float]


class YouTubeResult(TypedDict, total=False):
    """Metadata fields used from a yt-dlp search result."""

    id: str
    title: str
    uploader: str
    channel: str
    description: str
    duration: float
    duration_string: str
    view_count: Optional[int]


class ScoredYouTubeResult(YouTubeResult):
    """A YouTube result augmented with local ranking diagnostics."""

    rank_weight: float
    log_views: float
    title_boost: float
    artist_match_score: float
    keyword_boost: float
    matched_keywords: List[KeywordMatch]
    duration_score: float
    duration_cluster: Optional[Cluster]
    duration_cluster_fit: float
    final_score: float


@dataclass(frozen=True)
class DurationClusterConfig:
    """Tuning parameters for duration-cluster detection."""

    bin_width: int = 5
    window: int = 45
    max_clusters: int = 3
    min_duration: float = 120.0
    max_duration: float = 12 * 60.0
    relaxed_max_duration: float = 20 * 60.0
    minimum_samples: int = 3


DEFAULT_DURATION_CONFIG = DurationClusterConfig()


@dataclass(frozen=True)
class ScoringContext:
    """Shared inputs used to score each candidate result."""

    query_title: str
    query_artists: Sequence[str]
    preferred_keywords: Sequence[str]
    clusters: Sequence[Cluster]
    normalized_views: Sequence[float]
    result_count: int

console = Console()

KEYWORD_WEIGHTS: Dict[str, float] = {
    'original': 0.20,
    'official': 0.25,
    'remaster': 0.20,
    'extended': 0.10,
    'audio': 0.1,
    'radio edit': -0.1,
    'live': -0.50,
    'show': -1.00,
    'movie': -0.75,
    'pelicula': -0.75,
    'film': -0.50,
    'dvd': -0.35,
    'tutorial': -1.00,
    'lesson': -0.80,
    'concert': -0.50,
    'stage': -0.35,
    'karaoke': -0.75,
    'lyrics': 0.15,
    'video': -0.05,
}

PREFERRED_KEYWORD_WEIGHT = 0.20

SCORES: Dict[str, float] = {
    'rank_weight': 0.45,
    'log_views': 0.35,
    'title_boost': 1.2,
    'artist_boost': 1.5,
    'keyword_boost': 1.0,
    'duration_score': 0.8,
}

TITLE_NORMALIZATION_MAP = {
    "nothin": "nothing",
    "goin": "going",
    "shakin": "shaking",
    "lotta": "lot of",
    "aint": "ain't",
    "lets": "let's",
    "gonna": "going to",
    "wanna": "want to",
    "ya": "you",
    "u": "you",
    "ur": "your",
    "im": "i'm",
    "ive": "i've",
    "dont": "don't",
    "cant": "can't",
    "couldnt": "couldn't",
    "wouldnt": "wouldn't",
    "shouldnt": "shouldn't",
    "doesnt": "doesn't",
    "isnt": "isn't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "arent": "aren't",
    "didnt": "didn't",
    "havent": "haven't",
    "hasnt": "hasn't",
    "hadnt": "hadn't",
    "wont": "won't",
    "shes": "she's",
    "hes": "he's",
    "theyre": "they're",
    "theres": "there's",
    "whos": "who's",
    "whats": "what's",
    "youre": "you're",
    "weve": "we've",
}

TITLE_NORMALIZATION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(key) for key in TITLE_NORMALIZATION_MAP) + r")\b"
)
NON_WORD_PATTERN = re.compile(r"[^\w\s']")
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9]+")

NOISE_WORDS = {
    "album", "and", "audio", "best", "broadcast", "clean", "classic", "cover", "edit",
    "explicit", "feat", "featuring", "full", "ft", "high", "hq", "hd", "live", "lyrics",
    "mix", "mp4", "music", "official", "original", "performance", "produced", "prod",
    "promo", "quality", "radio", "recording", "remastered", "single", "track", "tv",
    "version", "versus", "video", "visualizer", "vs", "with"
}


def normalize_for_scoring(text: str, mode: Literal["artist", "title", "keyword"] = "title") -> str:
    """
    Normalize a YouTube title or artist string for consistent fuzzy matching.

    Modes:
    - 'title': aggressive cleaning, noise filtering, and slang normalization
    - 'artist': less aggressive, preserves inner structure but truncates 'The ' from beginning
    - 'keyword': like 'title', but skips noise filtering
    """
    if not text:
        return ""

    # Lowercase & strip accents
    s = unidecode(text.lower())

    # Replace slang / contractions
    s = TITLE_NORMALIZATION_PATTERN.sub(lambda match: TITLE_NORMALIZATION_MAP[match.group(0)], s)

    # Treat punctuation as a token boundary while preserving apostrophes.
    s = NON_WORD_PATTERN.sub(" ", s)

    # Normalize whitespace
    s = WHITESPACE_PATTERN.sub(" ", s).strip()

    # Special handling for artist: strip leading "the"
    if mode == "artist" and s.startswith("the "):
        s = s[4:]

    # Tokenize
    words = s.split()

    if mode == "title":
        words = [w for w in words if w not in NOISE_WORDS]
    elif mode == "artist":
        # Keep "the" inside name, only removed if leading
        pass
    elif mode == "keyword":
        # Keep everything (including noise)
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return " ".join(words)


def rescale_match_score(score: float, floor: float) -> float:
    """Map fuzzy-match noise at or below ``floor`` to zero."""
    if score <= floor:
        return 0.0
    return min(1.0, (score - floor) / (1.0 - floor))


def compact_match_text(text: str) -> str:
    """Remove formatting characters for punctuation-insensitive comparison."""
    return NON_ALPHANUMERIC_PATTERN.sub("", unidecode(text.lower()))


def get_artist_match_score(query_artists: Sequence[str], channel_name: str, title: str) -> float:
    """
    Return artist match score based on fuzzy match with uploader/channel/title.
    """
    norm_artists = [
        normalize_for_scoring(query_artist, mode="artist")
        for query_artist in query_artists
    ]
    norm_channel = normalize_for_scoring(channel_name, mode="artist")
    norm_title = normalize_for_scoring(title, mode="artist")
    norm_candidates = [norm_channel, norm_title]

    best_score = 0.0
    for norm_artist in norm_artists:
        if not norm_artist:
            continue
        for norm_candidate in norm_candidates:
            if not norm_candidate:
                continue
            score = partial_ratio(norm_artist, norm_candidate) / 100
            compact_artist = compact_match_text(norm_artist)
            compact_candidate = compact_match_text(norm_candidate)
            if len(compact_artist) >= 5:
                score = max(
                    score,
                    partial_ratio(compact_artist, compact_candidate) / 100,
                )
            best_score = max(best_score, score)
    return rescale_match_score(best_score, floor=0.75)


def get_title_match_score(query_title: str, candidate_title: str) -> float:
    """
    Compute a soft match score (0.0 to 1.0) between query title and candidate title.
    Uses fuzzy matching to allow for spelling variations, missing punctuation, etc.
    """
    query_title = normalize_for_scoring(query_title, mode="title")
    candidate_title = normalize_for_scoring(candidate_title, mode="title")

    if not query_title or not candidate_title:
        return 0.0

    # Fuzzy partial string match
    partial = partial_ratio(query_title, candidate_title) / 100
    token = token_sort_ratio(query_title, candidate_title) / 100

    # YouTube titles often omit apostrophes or use accent marks as apostrophes,
    # sometimes without a following space (for example, "Goin´On"). Comparing
    # compact forms makes those formatting differences free while retaining order.
    compact_query = compact_match_text(query_title)
    compact_candidate = compact_match_text(candidate_title)
    compact_partial = 0.0
    if len(compact_query) >= 8 and compact_candidate:
        compact_partial = partial_ratio(compact_query, compact_candidate) / 100

    return rescale_match_score(max(partial, token, compact_partial), floor=0.65)


def get_match_color(score: float) -> str:
    """Return a Rich color name for a normalized match score."""
    if score >= 0.95:
        return "green"
    if score >= 0.85:
        return "yellow"
    return "red"


def keyword_pattern(keyword: str) -> str:
    """Build a whole-word regex for a normalized keyword."""
    if keyword == "remaster":
        return r"\bremaster(?:ed)?\b"
    if keyword == "lyrics":
        return r"\blyrics?\b"
    return rf"\b{re.escape(keyword)}\b"


def extract_keyword_score(
    title: str,
    preferred_keywords: Optional[Sequence[str]] = None,
    protected_texts: Optional[Sequence[str]] = None,
) -> Tuple[float, List[KeywordMatch]]:
    """Return the keyword contribution and the terms that produced it."""
    norm_title = normalize_for_scoring(title, mode="keyword")
    protected = " ".join(
        normalize_for_scoring(text, mode="keyword")
        for text in protected_texts or ()
    )
    score = 0.0
    matched: List[KeywordMatch] = []
    for keyword, weight in KEYWORD_WEIGHTS.items():
        norm_keyword = normalize_for_scoring(keyword, mode="keyword")
        pattern = keyword_pattern(norm_keyword)
        if weight < 0 and re.search(pattern, protected):
            continue
        if re.search(pattern, norm_title):
            score += weight
            matched.append((keyword, weight))
    for keyword in preferred_keywords or []:
        norm_keyword = normalize_for_scoring(keyword, mode="keyword")
        if norm_keyword and re.search(keyword_pattern(norm_keyword), norm_title):
            score += PREFERRED_KEYWORD_WEIGHT
            matched.append((f"preferred: {keyword}", PREFERRED_KEYWORD_WEIGHT))
    return score, matched


def extract_description_penalty(
    description: str,
    protected_texts: Optional[Sequence[str]] = None,
    excluded_keywords: Optional[Sequence[str]] = None,
) -> Tuple[float, List[KeywordMatch]]:
    """Extract unique negative version signals from a video description."""
    _, matches = extract_keyword_score(
        description,
        protected_texts=protected_texts,
    )
    excluded = set(excluded_keywords or ())
    penalties = [
        (f"description: {keyword}", weight)
        for keyword, weight in matches
        if weight < 0 and keyword not in excluded
    ]
    return sum(weight for _, weight in penalties), penalties


def trim_by_quantiles(x: np.ndarray, lo: float = 0.02, hi: float = 0.98) -> np.ndarray:
    """
    Keeps the middle [lo, hi] quantile range.
    This is multi-modal friendly: it doesn't assume a single center.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 10:
        return x  # too small to trim safely
    q_lo, q_hi = np.quantile(x, [lo, hi])
    return x[(x >= q_lo) & (x <= q_hi)]


def _cluster_from_members(
    members: np.ndarray,
    *,
    fallback_sigma: float,
    sigma_bounds: Tuple[float, float],
) -> Cluster:
    """Summarize duration samples as a center, spread, and member count."""
    center = float(np.mean(members))
    sigma = float(np.std(members, ddof=1)) if len(members) >= 4 else fallback_sigma
    sigma = float(np.clip(sigma, *sigma_bounds))
    return center, sigma, int(len(members))


def _add_tail_cluster_if_needed(
    durations: np.ndarray,
    clusters: List[Cluster],
    *,
    config: DurationClusterConfig,
    side: Literal["high", "low"],
) -> None:
    """Add a duration mode near one edge when the histogram misses it."""
    edge = float(np.max(durations) if side == "high" else np.min(durations))
    if side == "high":
        members = durations[durations >= edge - config.window]
    else:
        members = durations[durations <= edge + config.window]

    if len(members) < config.minimum_samples:
        return

    candidate = _cluster_from_members(
        members,
        fallback_sigma=40.0,
        sigma_bounds=(25.0, 150.0),
    )
    if all(abs(center - candidate[0]) > config.window / 2 for center, _, _ in clusters):
        clusters.append(candidate)


def _plausible_durations(
    durations: np.ndarray,
    config: DurationClusterConfig,
) -> np.ndarray:
    """Filter invalid and implausible durations, relaxing the upper bound if needed."""
    finite = durations[np.isfinite(durations) & (durations > 0)]
    plausible = finite[
        (finite >= config.min_duration) & (finite <= config.max_duration)
    ]
    if len(plausible) < 6:
        plausible = finite[
            (finite >= config.min_duration)
            & (finite <= config.relaxed_max_duration)
        ]
    return np.sort(trim_by_quantiles(plausible))


def _histogram_clusters(
    durations: np.ndarray,
    config: DurationClusterConfig,
) -> List[Cluster]:
    """Find duration modes from histogram peaks."""
    bins = np.arange(
        durations.min(),
        durations.max() + config.bin_width,
        config.bin_width,
    )
    counts, edges = np.histogram(durations, bins=bins)
    distance = max(1, int(config.window / config.bin_width))
    prominence = max(1, int(0.15 * counts.max()))
    peaks, _ = find_peaks(counts, distance=distance, prominence=prominence)

    clusters: List[Cluster] = []
    for peak in peaks:
        start = edges[peak]
        members = durations[
            (durations >= start) & (durations <= start + config.window)
        ]
        if len(members) >= 2:
            clusters.append(
                _cluster_from_members(
                    members,
                    fallback_sigma=40.0,
                    sigma_bounds=(25.0, 150.0),
                )
            )
    return clusters


def smart_duration_clusters_pure(
    durations: np.ndarray,
    config: DurationClusterConfig = DEFAULT_DURATION_CONFIG,
) -> List[Cluster]:
    """Find up to ``max_clusters`` plausible duration modes."""
    plausible = _plausible_durations(np.asarray(durations, dtype=float), config)
    if len(plausible) < config.minimum_samples:
        center = float(np.mean(plausible)) if len(plausible) else 240.0
        return [(center, 60.0, int(len(plausible)))]

    clusters = _histogram_clusters(plausible, config)
    _add_tail_cluster_if_needed(plausible, clusters, config=config, side="high")
    _add_tail_cluster_if_needed(plausible, clusters, config=config, side="low")

    if not clusters:
        clusters = [
            _cluster_from_members(
                plausible,
                fallback_sigma=70.0,
                sigma_bounds=(30.0, 160.0),
            )
        ]

    clusters.sort(key=lambda cluster: cluster[2] / cluster[1], reverse=True)
    return clusters[:config.max_clusters]


def gaussian_score(value: float, center: float, sigma: float) -> float:
    """
    Standard Gaussian proximity score.

    Why Gaussian?
        Smooth penalty for deviation from cluster center.
        Falls off quadratically, not linearly.

    sigma:
        Learned from cluster spread (adaptive).
        Clipped earlier to prevent instability.
    """
    return float(np.exp(-((value - center) ** 2) / (2.0 * sigma ** 2)))


def duration_score_multicluster_pure(
    duration_s: Optional[float],
    clusters: Sequence[Cluster],
) -> float:
    """
    Final duration score.

    Hard rejection rules (duration-only, no title logic):

    < 90 seconds:
        Almost certainly clip/preview.

    > 20 minutes:
        Almost certainly DJ set / podcast.

    Otherwise:
        Score against the best-fitting cluster.
    """

    if duration_s is None:
        return 0.0

    d = float(duration_s)

    if d <= 0:
        return 0.0

    if d < 90.0:
        return 0.0

    if d > 20 * 60.0:
        return 0.0

    if not clusters:
        return 0.0

    return max(
        gaussian_score(d, center, sigma)
        for (center, sigma, _) in clusters
    )


def best_matching_cluster(
    duration_s: Optional[float],
    clusters: Sequence[Cluster],
) -> Tuple[Optional[Cluster], float]:
    """
    Returns (best_cluster, fit) where fit is the gaussian proximity to that cluster.
    """
    if duration_s is None or not clusters:
        return None, 0.0

    d = float(duration_s)
    best = None
    best_fit = -1.0

    for center, sigma, count in clusters:
        if sigma <= 0:
            fit = 0.0
        else:
            fit = math.exp(-((d - center) ** 2) / (2.0 * sigma ** 2))
        if fit > best_fit:
            best_fit = fit
            best = (center, sigma, count)

    return best, float(best_fit)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """Convert optional external metadata to a finite float."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _view_count(result: YouTubeResult) -> int:
    """Return a valid nonnegative view count from external metadata."""
    return max(0, int(_coerce_float(result.get("view_count"))))


def _normalized_log_views(results: Sequence[YouTubeResult]) -> List[float]:
    """Normalize log-transformed view counts to the inclusive range 0..1."""
    if not results:
        return []
    values = np.log1p([_view_count(result) for result in results])
    minimum = float(np.min(values))
    spread = float(np.max(values) - minimum)
    if spread == 0.0:
        return [0.0] * len(results)
    return [float((value - minimum) / spread) for value in values]


def _combined_keyword_score(
    title: str,
    description: str,
    context: ScoringContext,
) -> Tuple[float, List[KeywordMatch]]:
    """Combine title signals with unique negative evidence from the description."""
    protected_texts = [context.query_title, *context.query_artists]
    score, matches = extract_keyword_score(
        title,
        context.preferred_keywords,
        protected_texts,
    )
    title_penalties = [keyword for keyword, weight in matches if weight < 0]
    description_score, description_matches = extract_description_penalty(
        description,
        protected_texts,
        title_penalties,
    )
    return score + description_score, matches + description_matches


def _score_result(
    result: YouTubeResult,
    index: int,
    context: ScoringContext,
) -> ScoredYouTubeResult:
    """Score one result and attach its component diagnostics."""
    title = str(result.get("title") or "")
    channel = str(result.get("uploader") or result.get("channel") or "")
    duration = _coerce_float(result.get("duration"))
    rank_weight = (
        1.0
        if context.result_count <= 1
        else 1.0 - (index / (context.result_count - 1))
    )
    artist_match = get_artist_match_score(context.query_artists, channel, title)
    title_match = get_title_match_score(context.query_title, title)
    keyword_score, matched_keywords = _combined_keyword_score(
        title,
        str(result.get("description") or ""),
        context,
    )
    duration_score = duration_score_multicluster_pure(duration, context.clusters)
    duration_cluster, cluster_fit = best_matching_cluster(duration, context.clusters)
    final_score = (
        rank_weight * SCORES["rank_weight"]
        + context.normalized_views[index] * SCORES["log_views"]
        + title_match * SCORES["title_boost"]
        + artist_match * SCORES["artist_boost"]
        + keyword_score * SCORES["keyword_boost"]
        + duration_score * SCORES["duration_score"]
    )
    return cast(
        ScoredYouTubeResult,
        cast(object, {
            **result,
            "rank_weight": rank_weight,
            "log_views": context.normalized_views[index],
            "title_boost": title_match,
            "artist_match_score": artist_match,
            "keyword_boost": keyword_score,
            "matched_keywords": matched_keywords,
            "duration_score": duration_score,
            "duration_cluster": duration_cluster,
            "duration_cluster_fit": cluster_fit,
            "final_score": final_score,
        }),
    )


def score_youtube_results(
    results: Sequence[YouTubeResult],
    query_title: str,
    query_artists: Sequence[str],
    preferred_keywords: Optional[Sequence[str]] = None,
) -> List[ScoredYouTubeResult]:
    """Score and rank candidate videos from strongest to weakest match."""
    durations = [_coerce_float(result.get("duration")) for result in results]
    valid_durations = np.asarray([duration for duration in durations if duration > 0])
    context = ScoringContext(
        query_title=query_title,
        query_artists=query_artists,
        preferred_keywords=preferred_keywords or (),
        clusters=smart_duration_clusters_pure(valid_durations),
        normalized_views=_normalized_log_views(results),
        result_count=len(results),
    )
    scored = [
        _score_result(result, index, context)
        for index, result in enumerate(results)
    ]
    return sorted(scored, key=lambda result: result["final_score"], reverse=True)


def sort_scored_results(
    scored: Sequence[ScoredYouTubeResult],
    *,
    descending: bool = True,
) -> List[ScoredYouTubeResult]:
    """Return scored results in the requested score direction."""
    return sorted(
        scored,
        key=lambda result: result["final_score"],
        reverse=descending,
    )


def _keyword_text(matches: Sequence[KeywordMatch]) -> Text:
    """Render keyword contributions with positive and negative colors."""
    text = Text()
    for keyword, weight in matches:
        prefix = "+" if weight > 0 else ""
        color = "bold green" if weight > 0 else "bold red"
        text.append(f" {prefix}{keyword}({weight}) ", style=color)
    return text


def _add_score_row(
    table: Table,
    label: str,
    value: float,
    weight_key: str,
    display: Literal["plain", "colored", "precise"] = "plain",
) -> None:
    """Add one weighted score component to a Rich table."""
    weight = SCORES[weight_key]
    precision = 6 if display == "precise" else 3
    formatted_value = f"{value:.{precision}f}"
    value_cell = (
        Text(formatted_value, style=f"bold {get_match_color(value)}")
        if display == "colored"
        else formatted_value
    )
    table.add_row(
        label,
        value_cell,
        f"{weight:.1f}",
        f"{weight * value:.{precision}f}",
    )


def _score_table(result: ScoredYouTubeResult) -> Table:
    """Build the score-component table for one result."""
    table = Table(show_header=True, box=box.SIMPLE, highlight=True)
    table.add_column("Component")
    table.add_column("Value", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Contribution", justify="right")
    _add_score_row(table, "Rank Position", result["rank_weight"], "rank_weight")
    _add_score_row(table, "Log(Views)", result["log_views"], "log_views")
    _add_score_row(
        table,
        "Artist Match",
        result["artist_match_score"],
        "artist_boost",
        "colored",
    )
    _add_score_row(
        table,
        "Title Match",
        result["title_boost"],
        "title_boost",
        "colored",
    )
    _add_score_row(table, "Keyword Boost", result["keyword_boost"], "keyword_boost")
    _add_score_row(
        table,
        "Duration Score",
        result["duration_score"],
        "duration_score",
        "precise",
    )
    table.add_row(
        "[bold]Total Score[/bold]",
        "",
        "",
        f"[bold]{result['final_score']:.3f}[/bold]",
    )
    return table


def _print_result_header(index: int, result: ScoredYouTubeResult) -> None:
    """Print identifying metadata and keyword diagnostics for one result."""
    video_url = f"https://www.youtube.com/watch?v={result.get('id', '')}"
    cluster = result["duration_cluster"]
    console.rule(f"[bold cyan]Result #{index} — {result['final_score']:.3f}")
    console.print(f"[bold]Title:[/bold] {result.get('title', '?')}")
    console.print(f"[bold]Channel:[/bold] {result.get('channel', '?')}")
    console.print(f"[bold]Uploader:[/bold] {result.get('uploader', '?')}")
    console.print(f"[bold]URL:[/bold] [blue]{video_url}[/blue]")
    console.print(
        f"[bold]Duration:[/bold] {result.get('duration', '?')} sec "
        f"(Cluster: {cluster if cluster else 'None'})"
    )
    console.print(f"[bold]View Count:[/bold] {result.get('view_count', '?')}")
    if result["matched_keywords"]:
        console.print(
            "[bold]Matched Keywords:[/bold]",
            _keyword_text(result["matched_keywords"]),
        )


def print_scored_youtube_results(
    scored: Sequence[ScoredYouTubeResult],
    descending_score: bool = True,
) -> None:
    """Print ranked results and their score components."""
    for index, result in enumerate(
        sort_scored_results(scored, descending=descending_score),
        start=1,
    ):
        _print_result_header(index, result)
        console.print(_score_table(result))


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Score YouTube search results for a track.")
    parser.add_argument("-q", "--quantity", type=int, default=30, help="Number of YouTube results to fetch")
    parser.add_argument("-t", "--track-name", required=True, help="Track title to search for")
    parser.add_argument(
        "-a",
        "--artist",
        action="append",
        required=True,
        help="Artist name(s). Pass multiple times for OR search"
    )
    parser.add_argument(
        "-x",
        "--extra-args",
        "--prefer",
        dest="preferred_keywords",
        action="append",
        help="Preferred title keyword used only for local ranking (e.g. 'original')",
        default=[]
    )
    parser.add_argument("-o", "--output-json", help="Output all the JSON data to this file")
    parser.add_argument(
        "-r",
        "--reversed",
        action="store_true",
        help="Print lowest-scoring results first",
    )
    return parser.parse_args()


def build_search_query(track_name: str, artists: Sequence[str], quantity: int) -> str:
    """Build an artist/title-only yt-dlp search expression."""
    artist_expression = " OR ".join(repr(artist) for artist in artists)
    return f"ytsearch{quantity}:({artist_expression}) {track_name!r}"


def fetch_youtube_results(
    track_name: str,
    artists: Sequence[str],
    quantity: int = 30,
) -> List[YouTubeResult]:
    """Fetch flat YouTube search results, returning an empty list on failure."""
    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "extract_flat": "in_playlist",  # Do not resolve full metadata (faster)
    }
    search_query = build_search_query(track_name, artists, quantity)
    console.print(f"[bold]Search query:[/bold] {search_query}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            raw_results = ydl.extract_info(search_query, download=False, process=True)
    except DownloadError as error:
        console.print(f"[bold red]YouTube search failed:[/bold red] {error}")
        return []

    if not isinstance(raw_results, dict):
        return []

    results: List[YouTubeResult] = []
    for entry in raw_results.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        result = cast(YouTubeResult, cast(object, dict(entry)))
        duration = _coerce_float(result.get("duration"))
        if duration > 0:
            result["duration"] = duration
            result["duration_string"] = str(timedelta(seconds=duration)).lstrip("0:")
        results.append(result)
    return results


def main() -> int:
    """Run the command-line workflow and return a process exit status."""
    args = parse_args()
    results = fetch_youtube_results(
        track_name=args.track_name,
        artists=args.artist,
        quantity=args.quantity,
    )
    scored = score_youtube_results(
        results,
        args.track_name,
        args.artist,
        preferred_keywords=args.preferred_keywords,
    )
    print_scored_youtube_results(scored, descending_score=not args.reversed)

    if args.output_json:
        try:
            with open(args.output_json, "w", encoding="utf-8") as output_file:
                simplejson.dump(scored, output_file, indent=4, sort_keys=True)
        except OSError as error:
            console.print(f"[bold red]Could not write JSON output:[/bold red] {error}")
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
