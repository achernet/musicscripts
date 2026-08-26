#!/usr/bin/env python3
"""Build and play randomized playlists from the user's music library."""

import argparse
import logging
import math
import os
import platform
import random
import re
import select
import signal
import shutil
import subprocess
import sys
import time
import traceback

if os.name == "nt":
    import msvcrt
else:
    import termios
    import tty

import tqdm
from path import Path

PLAY_PATH = shutil.which("play.exe" if os.name == "nt" else "play")

DEFAULT_SONGS_PATH = Path("~/Music").expanduser()
RANDOM_SONGS_PATH = DEFAULT_SONGS_PATH / "random"
JAZZ_PATH = DEFAULT_SONGS_PATH / "jazz"
CLASSICAL_PATH = DEFAULT_SONGS_PATH / "classical"
LOG = logging.getLogger("play_songs")

# If Ctrl+C is pressed twice in this time period, it will quit.
QUIT_INTERVAL = 0.33

# Handle jazz options using bit flags.
JAZZ_RANDOM_BIT = 1
JAZZ_FOLDER_BIT = 2

# Global last interrupt timestamp
last_keyboard_interrupt_timestamp = 0


class RandomSorter:
    """Order songs randomly, optionally biased by filesystem timestamps."""

    USE_ATIME = False
    USE_MTIME = False
    DEFAULT_RANDOM_FACTOR = 0.25
    DEFAULT_RANDOM_OFFSET = 0.75
    REVERSE = False

    def __init__(self, use_atime=None, use_mtime=None, factor=None, offset=None, reverse=None):
        # ``None`` means "use the class default"; explicit False and 0 are
        # valid settings and must not be replaced by those defaults.
        self.use_atime = self.USE_ATIME if use_atime is None else use_atime
        self.use_mtime = self.USE_MTIME if use_mtime is None else use_mtime
        self.factor = self.DEFAULT_RANDOM_FACTOR if factor is None else factor
        self.offset = self.DEFAULT_RANDOM_OFFSET if offset is None else offset
        self.reverse = self.REVERSE if reverse is None else reverse

    def sort_key(self, song: Path) -> float:
        now = time.time()
        atime_diff = abs(now - song.atime)
        atime_val = math.log1p(atime_diff)
        mtime_diff = abs(song.mtime - now)
        mtime_val = math.log1p(mtime_diff)
        LOG.debug("atime: (%s, %s), mtime: (%s, %s)", atime_diff, atime_val, mtime_diff, mtime_val)
        if self.use_atime:
            if self.use_mtime:
                return 0.5 * (atime_val + mtime_val)
            return atime_val
        return mtime_val

    def random_sort_by_time(self, songs):
        """Shuffle songs, optionally retaining some timestamp-based order."""
        songs = list(songs)
        if not self.use_atime and not self.use_mtime:
            random.shuffle(songs)
            return songs
        if not songs:
            return []
        weighted_songs = []
        for i, song in enumerate(sorted(songs, key=self.sort_key)):
            weight = (self.factor * random.random() + self.offset) * i / len(songs)
            weighted_songs.append((song, weight))
        ordered = sorted(weighted_songs, key=lambda item: item[1], reverse=self.reverse)
        return [song for song, _weight in ordered]


def list_all_songs(recurse=False):
    """Return MP3s from the primary random-music directory."""
    if recurse:
        return list(RANDOM_SONGS_PATH.walkfiles("*.mp3"))
    return RANDOM_SONGS_PATH.files("*.mp3")


def songs_matching(pattern, *, include=True):
    """Select random-folder songs whose basenames match (or do not match)."""
    return [
        song
        for song in list_all_songs(recurse=False)
        if bool(re.search(pattern, song.basename(), flags=re.I)) is include
    ]


# noinspection SpellCheckingInspection
def gather_songs(sorter, old=False, kfma=False, trance=None, jazz=0, classical=False):
    """Build the requested playlist and return it in randomized order.

    Options are evaluated in priority order because the historical CLI permits
    several playlist switches to be supplied at once.
    """
    blacklist = "|".join([
        "3\\s+doors", "\\bcopland", "\\baqua(?!rius)", "\\blambo", "\\bsonata", "\\b182", "\\bjohn\\s+prine",
        "\\bbonnie", "\\bchevelle", "\\bantwoord", "\\bfortress", "\\bfoster", "\\bfrou", "\\bwalder", "\\bbizkit",
        "\\bcandyman", "\\bludacris", "\\bmanring", "\\bmussorg", "\\bmoby", "\\bminaj", "\\bno\\s+doubt",
        "\\bschilling", "pharrell", "mudd", "rossini", "shadowfax", "\\bsting", "c\\.mp3", "\\bstraits", "\\bweezer",
        "\\bbone\\s+thugs", "\\bwhite\\s+town", "willya", "henley", "lennox|eurythmics", "the\\s+glow", "metallica",
        "snoop\\s+dogg", "phil\\s+cunningham", "take\\s+on\\s+me", "dave\\s+brubeck", "godsmack", "enya", "posner",
        "lennox", "millionaire", "hiroshima", "tool\\s+\\W\\s+(?:schism|sober)", "sevendust", "disturbed", "football",
        "slightly\\s+stoopid", "\\bwildwest", "\\bvivaldi", "\\bfrankie\\s+goes\\s+to\\s+hollywood", "\\brebelution",
        "human\\s+league", "\\bspyro\\s+gyra", "\\btchaikovsky", "most\\s+iconic\\s+classical", "fugees",
        "simple\\s+minds", "\\bpegboard", "tears\\s+for\\s+fears", "al\\s+di\\s*meola", "staind", "50\\s+cent",
        "wallflowers", "new\\s+radicals", "38\\s+special", "pet\\s+shop\\s+boys", "(?:bob|boney)\\s+james",
        "acoustic\\s+alchemy", "dave\\s+koz", "korn", "haddaway", "bon\\s+jovi", "george\\s+benson",
        "drowning\\s+pool", "mission\\s+impossible", "michelle\\s+branch", "britney\\s+spears", "paper\\s+planes",
        "mellencamp", "kenny\\s+loggins", "lenny\\s+kravitz", "harvey\\s+danger", "robert\\s+miles",
        "everlast(?!ing)", "coolio", "claptone", "energy\\s+52", "yazoo", "dj\\s+sammy", "joe\\s+jackson",
        "corona.*?rhythm", "toto.*?africa", "deadmau5", "foreigner", "sade[ -]+", "flock\\s+(?:of\\s+)?seagulls",
        "outkast", "tom\\s+petty", "everything\\s+but\\s+the\\s+girl", "depeche\\s+mode", "nightnoise", "etheridge",
        "la\\s+bouche", "\\bprince\\W{1,4}", "ozzy", "rod\\s+stewart", "chamillionaire", "coldplay", "p\\.o\\.d",
        "kylie\\s+minogue", "kungs", "rage\\s+against\\s+the\\s+machine", "madonna", "ashanti", "delerium",
        "system\\s+of\\s+a\\s+down", "black\\s+eyed\\s+peas", "franz\\s+ferdinand", "(?:nine|9)\\s+inch\\s+nails",
        "ac\\W+dc\\W+", "u2\\W+", "cher.*?believe", "(the\\s+)?police\\W+", "soundgarden", "quiet\\s+riot",
        "notorious\\s+[big.]+", "luniz", "green\\s+day", "cypress\\s+hill", "bush", "ace\\s+of\\s+base",
        "\\blush\\s+", "judas\\s+priest", "rob\\s+zombie", "faithless", "paul\\s+van\\s+dyk", "rammstein",
        "\\bbeastie\\s+boys", "bloodfury", "faith\\s+no\\s+more", "soft\\s+cell", "schonherz", "cee\\s*lo\\s+green",
        "\\blocal\\s+h\\W", "\\bnirvana", "\\boffspring", "\\blinkin\\s+park", "\\bred\\s+hot\\s+chill?i\\s+peppers",
        "\\bevanescence", "shiny\\s+toy\\s+guns", "\\bpowerman\\s+5000", "\\balice\\s+in\\s+chains",
        "\\bcollective\\s+soul", "\\bstardust", "\\bpeter\\s+white",
        "\\bwill(?:iam)?\\s+ackerman", "pat\\s+benatar", "\\bwhite\\s+zombie", "stone\\s+temple\\s+pilots",
        "\\bsimone\\s+vitullo", "kings\\s+of\\s+leon", "van\\s+halen", "paul\\s+hardcastle", "jazzmasters",
        "\\binterior\\W*hot\\s+beach", "smashing\\s+pumpkins", "andre\\s+nickatina", "\\bverve\\b", "\\bnils\\b",
        "\\bbilly\\s+idol", "duran\\s+duran", "\\bseal\\b", "gorillaz", "counting\\s+crows", "adiemus", "ramones",
        "\\bsavage\\s+garden", "\\bmighty\\s+bosstones", "\\bcure\\s+\\W\\s+", "\\bradiohead",
        "(\\b(mister|mr)\\W{1,4}){2}", "\\b(2|two)\\s+unlimited", "sheryl\\s+crow", "(3|three)\\s+days\\s+grace",
        "mark\\s+morrison", "george\\s+strait", "gwen\\s+stefani", "foo\\s+fighters", "\\bprodigy", "\\bmac\\s+dre",
        "papa\\s+roach", "clapton\\W{1,4}(change(\\s+the)?\\s+world|tears\\s+(in\\s+)?heaven)", "\\bkenny\\s+g",
        "phil\\s+collins", "bruce\\s+hornsby", "fatboy\\s+slim", "meredith\\s+brooks", "simply\\s+red",
        "\\bkillers\\W+", "\\bkittie", "weeknd", "\\bfitz.*?tantrums\\W+", "\\blevel\\s+(42|forty\\s+two)\\W",
        "paula\\s+cole", "deep\\s+blue\\s+something", "dragostea", "(\\b(numa)\\s{1,2}){2,}", "bob\\s+marley",
        "george\\s+winston", "mark\\s+isham", "(r\\.e\\.m|rem)\\W+", "andrew\\s+rayel", "michael\\s+jackson",
        "(21|twenty\\W{1,4}one)\\s+pilots", "\\bchumbawamba", "(t\\.l\\.c|tlc)\\W+", "portugal.*?man\\W+",
        "\\bnickelback", "matchbox\\s+(20|twenty)", "\\bmontell\\s+jordan", "\\bkid\\s+rock", "\\binxs\\W+",
        "\\btrain\\W{1,4}drops\\s+of\\s+jupiter", "\\balex\\s+de\\s+grassi", "\\bmarkus\\s+schulz", "\\bsponge",
        "\\bzodiac\\s+(\\w+)\\s+theme", "\\bkid\\s+cudi", "\\bti\\S{1,3}sto\\b", "\\bdarude\\W+", "\\bdanzig",
        "\\bsublime\\W+", "david\\s+bowie\\W+(?!space\\s+oddity)", "\\beminem", "\\balice\\s+deejay", "\\brihanna",
        "\\bzz\\s+top", "\\bneil\\s+young\\W{1,4}rockin", "\\bstevie\\s+nicks", "nicki\\s+minaj", "\\bsaliva",
        "\\bseether", "\\btoby\\s+keith", "\\bstar\\s*trek", "\\binner\\s+circle", "cars\\W{1,4}drive", "\\bsixpence",
        "\\bthird\\s+eye\\s+blind", "john\\s+williams", "\\bmazzy\\s+star", "norah\\s+jones", "\\braceend",
        "\\bnelly", "justin\\s+timberlake", "\\bjonas", "\\b(5|five)\\s+finger\\s+death\\s+punch",
        "\\bslipknot", "\\bluther\\s+vandross", "\\bthorogood", "\\bbrian\\s+culbertson", "\\bsteve\\s+winwood",
        "\\bfinal\\s+countdown", "\\bfrank\\s+black", "\\bmegadeth", "\\bmodest\\s+mouse", "\\bkeane",
        "\\bgin\\s+blossoms", "\\bmarilyn\\s+manson", "\\blive\\W{1,4}lightning\\s+crashes", "\\buman",
        "\\bfreak\\s+nasty", "\\bphoenix\\W{1,4}1901", "\\bpostal\\s+service", "\\bstone\\s+sour", "kirk\\s+whalum",
        "\\bnajee", "lady\\s+gaga", "\\bbilly\\s+currington", "\\bmen\\s+at\\s+work", "john\\s+cougar", "katy\\s+perry",
        "\\bricky\\s+martin", "larry\\s+coryell", "david\\s+sanborn", "\\bdaryl\\s+hall\\W{1,4}john\\s+oates\\b",
        "\\bbloodhound\\s+gang", "\\bjewel\\W+", "\\bprobspot", "\\bnatalie\\s+imbruglia", "\\bnatalie\\s+merchant",
        "\\benvio\\W+", "\\baly\\W{1,4}fila", "\\barmin\\s+van\\s+buuren", "\\bjohn\\s+(00\\s+)?fleming",
        "\\bice\\s+cube", "\\bkim\\s+waters", "\\bserj\\s+tankian", "spectrasoul", "\\bsean\\s+tyas", "\\bfastball",
        "\\bsuper8", "\\bgeorge\\s+michael", "\\bqueens\\s+(of\\s+the\\s+)?stone\\s+age", "\\batb",
        "\\bshania\\s+twain", "\\brichard\\s+elliot", "\\bdutch\\s+force", "\\bdope\\W{1,4}", "\\bsugar\\s+ray",
        "\\bsurvivor\\W{1,4}eye\\s+(of\\s+the\\s+)?tiger", "\\bnoemi", "\\bellie\\s+goulding", "\\bart\\s+of\\s+noise",
        "\\btaragana\\s+pyjarama", "\\bdesree", "\\bbenny\\s+benassi", "\\bkaskade", "\\b(b\\.b\\.e|bbe)\\W+",
        "\\bcosmic\\s+gate", "\\bpaul\\s+oakenfold", "\\balex\\s+(m\\.o\\.r\\.p\\.h|morph)\\W+",
        "\\bsir\\s+mix\\Wa\\Wlot", "\\bbinary\\s+finary", "\\bpush\\W*(?:the)\\W*legacy", "\\bdan\\s+stone",
        "\\bangoscia", "\\bjohn\\s+o\\W*callaghan", "\\bbryan\\s+kearney", "\\balt\\W*f4", "\\bandy\\s+tau",
        "\\bsean\\s+truby", "\\btaylor\\s+swift", "\\bcranberries", "\\bgarth\\s+brooks", "\\bjohn\\s+summit",
        "\\bjan\\s+hammer", "\\b(a\\s+)?perfect\\s+circle", "\\bilan\\s+bluestone", "\\braving\\s+lunatics",
        "\\bbeyonc(e|\xe9)\\W+", "\\bmanuel\\s+le\\s+saux", "\\bjay\\W{1,4}z\\W+", "\\bmonogato", "\\bdavid\\s+guetta",
        "\\btaylan", "\\bbrooks\\W{1,4}dunn", "\\bgarbage", "\\bblank\\W{1,4}jones", "\\bsun\\s+decade",
        "\\bsunlounger", "\\bwill\\s+smith", "\\bppk", "\\bbarthezz", "\\bferry\\s+corsten", "\\bsubtronics",
        "\\bsam\\s+laxton", "\\bchicane", "\\bdriftmoon", "\\bdaxson", "\\bgrum\\W{1,4}u", "\\ballen\\W{1,4}envy",
        "\\bliz\\s+story", "\\bgrouplove", "\\bestiva", "\\bt(e|\xe9)l(e|\xe9)popmusik", "\\btim\\s+bowman",
        "\\btransatlantic", "\\bcannons\\W{1,4}fire", "\\bpeter\\s+bjorn", "(four|4)\\s+non\\s+blondes",
        "\\bkoyah", "\\bunbeat", "\\bsash(!|a)\\W+", "\\bmary\\s+(j\\.?)?\\s+blige\\W+", "\\bsarah\\s+mclachlan",
        "\\bexolight", "\\bmark\\s+dior", "\\bkiyoi", "\\bmaywave", "\\bfaruk\\s+sabanci", "\\bgiuseppe\\s+ottaviani",
        "\\bthrillseekers", "\\bfactor\\s+b\\W{1,4}", "\\bstoneface", "\\bfictivision", "\\brank\\s+(1|one)\\W{1,4}",
        "\\bjan\\s+blomqvist", "\\btinlicker", "\\bdigital\\s+department", "\\bmark\\s+pledger", "\\bjaki\\s+song",
        "\\bhazem\\s+beltagui", "\\bandrew\\s+dream", "\\bdreamy\\W+", "\\barnej", "\\bcold\\s+blue", "\\barty",
        "\\bjames\\s+dymond", "\\bnitrous\\s+oxide", "\\bkandi\\W{1,4}", "\\bscott\\s+cossu", "\\bspin\\s+doctors",
        "\\bprincess\\s+superstar", "\\bsolarstone", "\\bvampire\\s+weekend", "\\bdroopy\\W{1,4}ate",
        "\\brandy\\s+newman\\W{1,4}walk\\s+to\\s+work", "\\bdido", "\\bnaked\\s+eyes", "\\bblondie", "\\baalto",
        "\\bmauro\\s+picotto", "\\blost\\s+witness", "\\bnu\\s+nrg", "\\bgareth\\s+emery", "\\blange\\W{1,4}",
        "\\bmichael\\s+hedges", "\\bdarol\\s+anger", "\\bsunny\\s+lax", "\\bferry\\s+tayle", "\\bproclaimers",
        "\\bblues\\s+traveler", "\\btrisha\\s+yearwood", "\\bwhite\\s+stripes", "\\bja\\s+rule",
        "\\bfine\\s+young\\s+cannibals", "\\bcan\\s+you\\s+feel\\s+(the\\s+)?love\\s+tonight",
        "\\bguns\\s+(n|and|&)\\s+roses", "\\bchris\\s+botti", "\\bgregg\\s+karukas", "\\bwill\\s+atkinson",
        "\\bwarren\\s+g\\.?\\W{1,4}regulate", "\\bwill\\s+rees", "\\benigma\\s+state", "\\bgouryella",
        "\\bamber\\W{1,4}this\\s+is\\s+your\\s+night", "\\benigma\\W{1,4}", "\\bmike\\s+nichol", "\\bevbointh",
        "\\bmasters\\W{1,4}nickson", "\\bkaymak\\W{1,4}mannix", "\\byork\\W{1,4}reachers", "\\bsolid\\s+sleep",
        "\\belectrovoya", "\\blostly", "\\byoung\\s+parisians", "\\bmark\\W{1,4}lukas", "\\bralphie\\s+b\\.?",
        "\\bapple\\s+one", "\\bfady\\s+(n|and|&|x)\\s+mina", "\\bafanasiev", "\\bandrea\\s+ribeca", "\\bafternova",
        "\\bsylent\\s+rain", "\\binsigma", "\\bglenn\\s+miller", "\\bpaul\\s+whiteman", "\\bkendrick\\s+lamar",
        "\\bsnap\\W{1,4}rhythm", "\\bbananarama",
    ])
    kfma_whitelist = "|".join([
        "3\\+doors", "blink.*?182", "chevelle", "disturbed", "drowning\\s+pool", "foster\\s+(?:the)?\\s*people",
        "godsmack", "harvey\\s+danger", "korn", "lenny\\s+kravitz", "limp\\s+bizkit", "paper\\s+planes",
        "metallica", "puddle\\s+of\\s+mudd", "sevendust", "staind", "tool", "weezer", "everlast(?!ing)", "ozzy",
        "p\\.o\\.d", "rage\\s+against\\s+the\\s+machine", "system\\s+of\\s+a\\s+down", "franz\\s+ferdinand",
        "(?:nine|9)\\s+inch\\s+nails", "ac\\W+dc\\W+", "bush", "cypress\\s+hill", "green\\s+day", "soundgarden",
        "rob\\s+zombie", "rammstein", "beastie\\s+boys", "faith\\s+no\\s+more", "offspring", "nirvana",
        "linkin\\s+park", "red\\s+hot\\s+chill?i\\s+peppers", "evanescence", "shiny\\s+toy\\s+guns",
        "powerman\\s+5000", "alice\\s+in\\s+chains", "white\\s+zombie", "stone\\s+temple\\s+pilots",
        "smashing\\s+pumpkins", "gorillaz", "ramones", "radiohead", "(3|three)\\s+days\\s+grace", "danzig",
        "foo\\s+fighters", "papa\\s+roach", "\\bkillers\\W+", "\\bkittie", "nickelback", "kid\\s+rock",
        "sublime", "\\bsponge", "\\bsaliva", "\\bseether", "\\b(5|five)\\s+finger\\s+death\\s+punch", "slipknot",
        "\\bmegadeth", "\\bmodest\\s+mouse", "\\bmarilyn\\s+manson", "\\bserj\\s+tankian",
        "\\bqueens\\s+(of\\s+the\\s+)?stone\\s+age", "\\bdope\\W{1,4}", "\\b(a\\s+)?perfect\\s+circle",
        "\\bgarbage", "\\bwhite\\s+stripes", "\\bguns\\s+(n|and|&)\\s+roses",
    ])
    jazz_whitelist = "|".join([
        "bob\\s+james", "\\bfourplay", "boney\\s+james", "dave\\s+koz", "acoustic\\s+alchemy", "\\bnils\\b",
        "will(?:iam)?\\s+ackerman", "\\bshadowfax", "\\bschonherz", "phil\\s+cunningham", "paul\\s+hardcastle",
        "jazzmasters", "\\bnightnoise", "michael\\s+manring", "mark\\s+isham", "\\bkenny\\s+g", "\\bira\\s+stein",
        "\\binterior\\W{1,4}hot\\s+beach", "\\bhiroshima", "george\\s+winston", "george\\s+benson", "\\benya",
        "dave\\s+brubeck", "brian\\s+culbertson", "alex\\s+de\\s+grassi", "al\\s+di\\s*meola", "\\badiemus",
        "\\buman", "kirk\\s+whalum", "\\bnajee", "\\bkim\\s+waters", "\\brichard\\s+elliot", "\\bliz\\s+story",
        "\\btim\\s+bowman", "\\bscott\\s+cossu", "\\bmichael\\s+hedges", "\\bdarol\\s+anger",
        "\\bglenn\\s+miller", "\\bgregg\\s+karukas", "\\bchris\\s+botti", "\\bpeter\\s+white",
    ])
    trance_whitelist = "|".join([
        "\\bsean\\s+tyas", "\\barmin\\s+van\\s+buuren", "\\bprobspot", "\\benvio\\W+", "\\baly\\W{1,4}fila",
        "\\bjohn\\s+(00\\s+)?fleming", "spectrasoul", "\\bdj\\s+tab", "\\bti\\S{1,3}sto\\b",
        "\\batb", "\\bsean\\s+tyas", "\\bsuper8", "\\balice\\s+deejay", "\\bdarude\\W+", "andrew\\s+rayel",
        "\\bbloodfury", "\\bpaul\\s+van\\s+dyk", "\\bfaithless", "\\bdelerium", "\\bdeadmau5", "\\bpegboard",
        "\\brobert\\s+miles", "\\bclaptone", "\\benergy\\s+52", "\\bdj\\s+sammy", "\\bnoemi", "\\bmarkus\\s+schulz",
        "\\bdutch\\s+force", "\\bart\\s+of\\s+noise", "\\btaragana\\s+pyjarama", "\\bbenny\\s+benassi",
        "\\bkaskade", "\\b(b\\.b\\.e|bbe)\\W+", "\\bcosmic\\s+gate", "\\balex\\s+(m\\.o\\.r\\.p\\.h|morph)\\W+",
        "\\bpaul\\s+oakenfold", "\\bbinary\\s+finary", "\\bpush\\W*(?:the)\\W*legacy", "\\bdan\\s+stone",
        "\\bangoscia", "\\bjohn\\s+o\\W*callaghan", "\\bbryan\\s+kearney", "\\balt\\W*f4", "\\bandy\\s+tau",
        "\\bsean\\s+truby", "\\bjohn\\s+summit", "\\bilan\\s+bluestone", "\\braving\\s+lunatics", "\\btaylan",
        "\\bmanuel\\s+le\\s+vaux", "\\bmonogato", "\\bdavid\\s+guetta", "\\bppk", "\\bbarthezz", "\\bferry\\s+corsten",
        "\\bsubtronics", "\\bsam\\s+laxton", "\\bchicane", "\\bdriftmoon", "\\bdaxson", "\\bgrum\\W{1,4}u",
        "\\ballen\\W{1,4}envy", "\\bblank\\W{1,4}jones", "\\bsun\\s+decade", "\\bsunlounger", "\\bsash(!|a)\\W+",
        "\\bt(e|\xe9)l(e|\xe9)popmusik", "\\bestiva", "\\btransatlantic", "\\bkoyah", "\\bunbeat",
        "\\bsarah\\s+mclachlan\\W+world\\s+on\\s+fire", "\\bexolight", "\\bmark\\s+dior", "\\bkiyoi", "\\bmaywave",
        "\\bfaruk\\s+sabanci", "\\bgiuseppe\\s+ottaviani", "\\bthrillseekers", "\\bfactor\\s+b\\W{1,4}",
        "\\bstoneface", "\\bfictivision", "\\brank\\s+(1|one)\\W{1,4}", "\\bjan\\s+blomqvist", "\\btinlicker",
        "\\bdigital\\s+department", "\\bmark\\s+pledger", "\\bjaki\\s+song", "\\bkandi\\W{1,4}", "\\barty",
        "\\bhazem\\s+beltagui", "\\bandrew\\s+dream", "\\bdreamy\\W+", "\\barnej", "\\bcold\\s+blue",
        "\\bjames\\s+dymond", "\\bnitrous\\s+oxide", "\\bprincess\\s+superstar", "\\bsolarstone", "\\baalto",
        "\\bmauro\\s+picotto", "\\blost\\s+witness", "\\bnu\\s+nrg", "\\bgareth\\s+emery", "\\blange\\W{1,4}",
        "\\bsunny\\s+lax", "\\bferry\\s+tayle", "\\bwill\\s+rees", "\\benigma\\s+state", "\\bgouryella",
        "\\bgregg\\s+karukas", "\\bwill\\s+atkinson", "\\bmike\\s+nichol", "\\bevbointh", "\\bmasters\\W{1,4}nickson",
        "\\bkaymak\\W{1,4}mannix", "\\bmasters\\W{1,4}nickson", "\\bkaymak\\W{1,4}mannix", "\\byork\\W{1,4}reachers",
        "\\bsolid\\s+sleep", "\\belectrovoya", "\\blostly", "\\byoung\\s+parisians", "\\bmark\\W{1,4}lukas",
        "\\bralphie\\s+b\\.?", "\\bapple\\s+one", "\\bfady\\s+(n|and|&|x)\\s+mina", "\\bafanasiev",
        "\\bandrea\\s+ribeca", "\\bafternova", "\\bsylent\\s+rain", "\\binsigma",
    ])
    if trance is True:  # -T was passed in
        trance_songs = songs_matching(trance_whitelist)
        return sorter.random_sort_by_time(trance_songs)
    if trance:  # -t was passed in, possibly with users afterward
        trance_songs = []
        for username in trance:
            trance_songs.extend((DEFAULT_SONGS_PATH / username).files("*.mp3"))
        return sorter.random_sort_by_time(trance_songs)
    if classical:
        classical_songs = CLASSICAL_PATH.files("*.mp3")
        return sorter.random_sort_by_time(classical_songs)
    if old:
        old_songs = songs_matching(blacklist, include=False)
        return sorter.random_sort_by_time(old_songs)
    if kfma:
        kfma_songs = songs_matching(kfma_whitelist)
        return sorter.random_sort_by_time(kfma_songs)
    if jazz > 0:
        jazz_selection = []
        if jazz & JAZZ_RANDOM_BIT:
            random_jazz_songs = songs_matching(jazz_whitelist)
            jazz_selection.extend(random_jazz_songs)
        if jazz & JAZZ_FOLDER_BIT:
            jazz_folder_songs = list(JAZZ_PATH.walkfiles("*.mp3"))
            jazz_selection.extend(jazz_folder_songs)
        return sorter.random_sort_by_time(jazz_selection)
    return sorter.random_sort_by_time(list_all_songs(recurse=False))


def play_song_showing_console_output(song):
    """Play one song while forwarding SoX output; return True on Ctrl+R."""
    if PLAY_PATH is None:
        raise RuntimeError("SoX 'play' executable was not found on PATH")

    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(
        [PLAY_PATH, str(song)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,  # unbuffered, so we get characters immediately
        creationflags=creation_flags,
        start_new_session=os.name != "nt",
    )

    pending_output = bytearray()
    redraw_line = False

    def compact_progress(output, width):
        """Turn SoX's fixed-width meter into a terminal-sized status line."""
        match = re.match(
            rb"\s*In:\s*(\S+)\s+(\S+)\s+\[([^]]+)]",
            output,
        )
        if not match:
            return output[:width]

        percent, elapsed, remaining = match.groups()
        candidates = (
            b"Playing: " + percent + b"  " + elapsed + b"  remaining " + remaining,
            percent + b"  " + elapsed + b"  [" + remaining + b" left]",
            percent + b"  " + elapsed,
            percent,
        )
        status = next((candidate for candidate in candidates if len(candidate) <= width), candidates[-1])

        # A decoder warning can arrive on the current progress line.  Preserve
        # it as a normal line instead of hiding it with the meter fields.
        warning_at = output.find(b"/usr/bin/play WARN")
        if warning_at >= 0:
            status += b"\n" + output[warning_at:]
        return status[:width] if b"\n" not in status else status

    def flush_output(delimiter=b""):
        """Write one player output line without letting a CR line wrap."""
        nonlocal redraw_line
        output = bytes(pending_output)
        pending_output.clear()
        if sys.stdout.isatty() and (redraw_line or delimiter == b"\r"):
            width = max(1, shutil.get_terminal_size(fallback=(80, 24)).columns - 1)
            if redraw_line and width < 77:
                output = compact_progress(output, width)
            else:
                output = output[:width]
            # Reset and clear before drawing.  This is independent of where
            # SoX placed its carriage return and cannot leave stale text.
            sys.stdout.buffer.write(b"\r\033[2K" + output)
            if delimiter == b"\n":
                sys.stdout.buffer.write(delimiter)
        else:
            sys.stdout.buffer.write(output + delimiter)
        sys.stdout.flush()
        redraw_line = delimiter == b"\r"

    try:
        while True:
            if os.name != "nt" and sys.stdin.isatty():
                readers = [sys.stdin]
                if proc.stdout:
                    readers.append(proc.stdout)
                readable, _, _ = select.select(readers, [], [])
                if sys.stdin in readable:
                    key = os.read(sys.stdin.fileno(), 1)
                    if key == b"\x12":  # Ctrl+R
                        stop_player(proc)
                        return True
                if proc.stdout not in readable:
                    continue
            # Read single bytes to preserve carriage-return progress updates.
            chunk = proc.stdout.read(1) if proc.stdout else b""
            if not chunk and proc.poll() is not None:
                break

            if chunk in (b"\r", b"\n"):
                flush_output(chunk)
            elif chunk:
                pending_output.extend(chunk)
    except KeyboardInterrupt:
        stop_player(proc)
        raise
    finally:
        if pending_output:
            flush_output()
        if proc.stdout:
            proc.stdout.close()
        if proc.poll() is None:
            stop_player(proc)
    return False


def stop_player(proc):
    """Stop the audio player and reap it without leaving the terminal altered."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def restore_terminal(settings):
    """Restore terminal input and display state after interactive playback."""
    if settings is not None:
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, settings)
        except (OSError, termios.error):
            LOG.warning("Could not restore terminal settings", exc_info=True)
    if sys.stdout.isatty():
        # Clear player formatting, show the cursor, and leave the prompt on a clean line.
        sys.stdout.write("\033[0m\033[?25h\n")
        sys.stdout.flush()


def play_songs(
        sorter,
        songs=None,
        old=False,
        kfma=False,
        list_only=False,
        trance=None,
        jazz=0,
        classical=False,
        continue_play=False,
):
    """Build a playlist, then either return it or play it."""
    if songs:
        randomized_songs = sorter.random_sort_by_time(songs)
        if old or kfma or trance or jazz or sorter.use_atime or sorter.use_mtime or continue_play:
            randomized_songs.extend(
                gather_songs(
                    sorter=sorter,
                    old=old,
                    kfma=kfma,
                    trance=trance,
                    jazz=jazz,
                    classical=classical,
                )
            )
    else:
        randomized_songs = gather_songs(
            sorter=sorter,
            old=old,
            kfma=kfma,
            trance=trance,
            jazz=jazz,
            classical=classical,
        )
    if list_only:
        return randomized_songs
    if platform.system() == "Darwin" and Path("/Applications/eqMac.app").exists():
        subprocess.run(["open", "-g", "/Applications/eqMac.app"], check=False)
        subprocess.run(
            ["open", "/System/Applications/Utilities/Terminal.app"],
            check=False,
        )
    return play_randomized_songs(randomized_songs)


def play_randomized_songs(randomized_songs):
    """Play each song, supporting Ctrl+R restart and double-Ctrl+C exit."""
    global last_keyboard_interrupt_timestamp
    songs_played = []

    def _ctrl_c_handler(_sig, _frame):
        raise KeyboardInterrupt()

    # Temporarily override SIGINT so Python doesn’t instantly exit
    orig_handler = signal.getsignal(signal.SIGINT)
    terminal_settings = None
    if os.name != "nt" and sys.stdin.isatty():
        terminal_settings = termios.tcgetattr(sys.stdin.fileno())
        # Make Ctrl+R available immediately while retaining signal keys like Ctrl+C.
        tty.setcbreak(sys.stdin.fileno())
    signal.signal(signal.SIGINT, _ctrl_c_handler)

    try:
        for song in randomized_songs:
            hit_ctrl_c = False
            try:
                while play_song_showing_console_output(song):
                    print("\nRestarting song...")

                # timestamp update
                try:
                    song.utime((int(time.time()), int(song.mtime)))
                except OSError:
                    print(f"Error updating timestamps on {song}, skipping...")
                    traceback.print_exc()
                songs_played.append(song)

            except (KeyboardInterrupt, SystemExit):
                hit_ctrl_c = True

            if hit_ctrl_c or msvcrt_kbhit_ctrl_c():
                keyboard_interrupt_timestamp = time.time()

                # quit on double-tap
                if keyboard_interrupt_timestamp - last_keyboard_interrupt_timestamp < QUIT_INTERVAL:
                    print("Double Ctrl+C detected, quitting...")
                    break
                last_keyboard_interrupt_timestamp = keyboard_interrupt_timestamp
                continue

    finally:
        # restore original signal handler
        signal.signal(signal.SIGINT, orig_handler)
        restore_terminal(terminal_settings)

    return songs_played


def msvcrt_kbhit_ctrl_c():
    """Consume and detect Ctrl+C from the Windows console, if available."""
    if os.name != "nt":
        return False
    if msvcrt.kbhit():
        key = msvcrt.getwch()
        if key == '\x03':
            return True
    return False


def build_argument_parser():
    """Create the command-line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-d",
        "--debug",
        default=False,
        action="store_true",
        help="print debug messages to stderr",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="print info messages to stderr",
    )
    ap.add_argument(
        "-o",
        "--old",
        default=False,
        action="store_true",
        help="Play old music (1950s, 1960s, some 1970s)",
    )
    ap.add_argument(
        "-l",
        "--list",
        default=False,
        action="store_true",
        help="Only list songs, don't play them",
    )
    ap.add_argument(
        "-a",
        "--atime",
        default=False,
        action="store_true",
        help="Prefer latest accessed songs first",
    )
    ap.add_argument(
        "-A",
        "--atime-reverse",
        default=False,
        action="store_true",
        help="Prefer latest accessed songs last",
    )
    ap.add_argument(
        "-m",
        "--mtime",
        default=False,
        action="store_true",
        help="Prefer latest modified songs first",
    )
    ap.add_argument(
        "-M",
        "--mtime-reverse",
        default=False,
        action="store_true",
        help="Prefer latest modified songs last",
    )
    ap.add_argument(
        "-k",
        "--kfma",
        default=False,
        action="store_true",
        help="Play songs known to be in the KFMA playlist (alt rock, metal)",
    )
    trance_group = ap.add_mutually_exclusive_group()
    trance_group.add_argument(
        "-t",
        "--trance-users",
        default=None,
        nargs="*",
        help="Play trance music fetched for the given users (default: %(default)s)",
    )
    trance_group.add_argument(
        "-T",
        "--trance-random",
        default=False,
        action="store_true",
        help="Play trance songs from random that match the trance whitelist",
    )
    ap.add_argument(
        "-j",
        "--jazz-random",
        default=False,
        action="store_true",
        help="Play jazz songs from random that match the jazz whitelist",
    )
    ap.add_argument(
        "-J",
        "--jazz-folder",
        default=False,
        action="store_true",
        help="Play jazz from the Music/jazz folder",
    )
    ap.add_argument(
        "-c",
        "--classical",
        default=False,
        action="store_true",
        help="Play classical music",
    )
    ap.add_argument(
        "-rf",
        "--random-factor",
        default=RandomSorter.DEFAULT_RANDOM_FACTOR,
        type=float,
        help="The random sort factor (default: %(default)s)",
    )
    ap.add_argument(
        "-ro",
        "--random-offset",
        default=RandomSorter.DEFAULT_RANDOM_OFFSET,
        type=float,
        help="The random offset (default: %(default)s)",
    )
    ap.add_argument(
        "-R",
        "--reset-dates",
        default=False,
        action="store_true",
        help="Reset all dates back to the file birth time (assumes Darwin/MacOS and implies --list)",
    )
    ap.add_argument(
        "-C",
        "--continue-play",
        default=False,
        action="store_true",
        help="Continue playing random music after playing the given songs (default: %(default)s)",
    )
    ap.add_argument(
        "songs",
        nargs="*",
        type=Path,
        help="If given alone, play these songs only; if given with other playlist args, play these first.",
    )
    return ap


def main(argv=None):
    """Run the command-line interface."""
    ns = build_argument_parser().parse_args(argv)

    logging.root.addHandler(logging.StreamHandler())
    if ns.verbose:
        logging.root.setLevel(logging.INFO)
    if ns.debug:
        logging.root.setLevel(logging.DEBUG)

    sorter = RandomSorter(
        use_atime=ns.atime or ns.atime_reverse,
        use_mtime=ns.mtime or ns.mtime_reverse,
        factor=ns.random_factor,
        offset=ns.random_offset,
        reverse=ns.atime_reverse or ns.mtime_reverse,
    )
    songs_played = play_songs(
        sorter=sorter,
        songs=ns.songs,
        old=ns.old,
        kfma=ns.kfma,
        list_only=ns.list or ns.reset_dates,
        trance=ns.trance_users if ns.trance_users else ns.trance_random,
        jazz=JAZZ_FOLDER_BIT * bool(ns.jazz_folder) + JAZZ_RANDOM_BIT * bool(ns.jazz_random),
        classical=ns.classical,
        continue_play=ns.continue_play,
    )
    if ns.list:
        try:
            print("\n".join([s.basename() for s in songs_played]))
            sys.stdout.flush()
        except BrokenPipeError:
            sys.exit(0)
        return
    if ns.reset_dates:
        for song in tqdm.tqdm(songs_played, total=len(songs_played), desc="Resetting dates"):
            song_date = int(song.stat().st_birthtime)
            set_file_date = time.strftime("%m/%d/%Y %H:%M:%S", time.localtime(song_date))
            song.utime((song_date, song_date))
            subprocess.run(["SetFile", "-d", set_file_date, str(song)], check=True)
        print("Dates have been reset on {0} files!".format(len(songs_played)))


if __name__ == "__main__":
    main()
