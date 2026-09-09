"""Script detection and transliteration to a Latin representation.

This module runs FIRST in the screening pipeline, before normalization and
before any phonetic key is computed. The reason is specific rather than
stylistic: Double Metaphone is a set of English grapheme-to-phoneme rules. Hand
it "Иванов" and it walks the string looking for English digraphs, finds none it
recognises, and returns a key derived from nothing. The result is not an error,
it is a plausible-looking wrong answer, so every non-Latin name would quietly
fall out of its phonetic block and never be compared to its true match.

For Arabic the naive path is worse than useless. Arabic is an abjad: short
vowels are not written. A pure codepoint map turns محمد into the consonant
skeleton "mhmd", whose Double Metaphone key is MMT, while every conventional
romanization (Mohammed / Muhammad / Mohamed) keys to MHMT. The skeleton would
therefore be blocked *away* from the very names it should match. That is why
Arabic goes through a name lexicon first and only falls back to the character
table for tokens the lexicon does not know.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Final

from unidecode import unidecode

from src.exceptions import TransliterationError


class Script(str, Enum):
    LATIN = "LATIN"
    ARABIC = "ARABIC"
    CYRILLIC = "CYRILLIC"
    HAN = "HAN"
    OTHER = "OTHER"


class Method(str, Enum):
    IDENTITY = "identity"          # already Latin
    LEXICON = "lexicon"            # matched a curated name form
    TABLE = "table"                # deterministic codepoint map
    PINYIN = "pinyin"              # Han -> Hanyu Pinyin
    FALLBACK = "fallback"          # unidecode, last resort
    MIXED = "mixed"                # token-wise combination of the above


_SCRIPT_RANGES: Final[tuple[tuple[int, int, Script], ...]] = (
    (0x0041, 0x024F, Script.LATIN),      # Basic Latin + Latin-1/Extended-A/B
    (0x1E00, 0x1EFF, Script.LATIN),      # Latin Extended Additional
    (0x0400, 0x052F, Script.CYRILLIC),   # Cyrillic + Supplement
    (0x0600, 0x06FF, Script.ARABIC),     # Arabic
    (0x0750, 0x077F, Script.ARABIC),     # Arabic Supplement
    (0xFB50, 0xFDFF, Script.ARABIC),     # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF, Script.ARABIC),     # Arabic Presentation Forms-B
    (0x4E00, 0x9FFF, Script.HAN),        # CJK Unified Ideographs
    (0x3400, 0x4DBF, Script.HAN),        # CJK Extension A
    (0xF900, 0xFAFF, Script.HAN),        # CJK Compatibility Ideographs
)

# BGN/PCGN romanization for Russian, extended with the Ukrainian and Belarusian
# letters that appear in OFAC's Ukraine/Belarus programs. Digraph outputs (zh,
# kh, shch) matter: they are what make the Latin form phonetically faithful.
CYRILLIC_MAP: Final[dict[str, str]] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "і": "i", "ї": "yi", "є": "ye", "ґ": "g", "ў": "w",
}

# Arabic codepoint fallback. Emphatic consonants collapse onto their plain
# counterparts (ص -> s, ط -> t) because no romanization convention in the SDN
# list preserves the distinction, and ع / ء are dropped rather than rendered as
# an apostrophe, which would only add a token boundary that nobody agrees on.
ARABIC_CHAR_MAP: Final[dict[str, str]] = {
    "ا": "a", "آ": "a", "أ": "a", "إ": "i", "ب": "b", "ت": "t", "ث": "th",
    "ج": "j", "ح": "h", "خ": "kh", "د": "d", "ذ": "dh", "ر": "r", "ز": "z",
    "س": "s", "ش": "sh", "ص": "s", "ض": "d", "ط": "t", "ظ": "z", "ع": "",
    "غ": "gh", "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n",
    "ه": "h", "و": "w", "ي": "y", "ى": "a", "ة": "a", "ء": "", "ؤ": "u",
    "ئ": "i", "پ": "p", "چ": "ch", "ژ": "zh", "گ": "g", "ک": "k", "ی": "y",
    # Harakat are usually absent from SDN entries but appear in scraped feeds.
    "َ": "a", "ِ": "i", "ُ": "u", "ّ": "", "ْ": "",
    "ً": "an", "ٍ": "in", "ٌ": "un", "ـ": "",
}

# Curated Arabic name forms. These are the tokens the fallback table cannot
# vowel correctly, and they are also the highest-frequency tokens on the SDN
# list, so the lexicon buys most of the accuracy for a small table.
ARABIC_NAME_LEXICON: Final[dict[str, str]] = {
    "محمد": "muhammad", "محمود": "mahmud", "أحمد": "ahmad", "احمد": "ahmad",
    "علي": "ali", "حسن": "hasan", "حسين": "husayn", "عبد": "abd",
    "الله": "allah", "عبدالله": "abdullah", "عبد الله": "abdullah",
    "عبدالرحمن": "abdulrahman", "عبدالعزيز": "abdulaziz", "خالد": "khalid",
    "عمر": "umar", "عثمان": "uthman", "إبراهيم": "ibrahim", "ابراهيم": "ibrahim",
    "يوسف": "yusuf", "سعيد": "said", "فاطمة": "fatima", "عائشة": "aisha",
    "زينب": "zaynab", "مريم": "maryam", "سلمان": "salman", "فيصل": "faysal",
    "ناصر": "nasir", "طارق": "tariq", "رشيد": "rashid", "كريم": "karim",
    "مصطفى": "mustafa", "جمال": "jamal", "سامي": "sami", "وليد": "walid",
    "ياسر": "yasir", "زياد": "ziyad", "هشام": "hisham", "بشار": "bashar",
    "أسد": "asad", "قاسم": "qasim", "صالح": "salih", "عادل": "adil",
    "نبيل": "nabil", "فريد": "farid", "رامي": "rami", "ماهر": "mahir",
    "أنور": "anwar", "سليم": "salim", "أمين": "amin", "نور": "nur",
    "بكر": "bakr", "يعقوب": "yaqub", "إسماعيل": "ismail", "اسماعيل": "ismail",
    "داوود": "dawud", "سليمان": "sulayman", "هارون": "harun", "إدريس": "idris",
    "الحسيني": "husayni", "الشامي": "shami", "البغدادي": "baghdadi",
    "الدمشقي": "dimashqi", "العراقي": "iraqi", "المصري": "misri",
    "الجزائري": "jazairi", "اللبناني": "lubnani", "التكريتي": "tikriti",
    "الزهراني": "zahrani", "القحطاني": "qahtani", "العتيبي": "utaybi",
    "الشهري": "shahri", "الغامدي": "ghamdi", "الحربي": "harbi",
    "شركة": "sharikat", "مؤسسة": "muassasat", "مصرف": "masraf",
    "بنك": "bank", "تجارة": "tijara", "الدولية": "aldawliya",
}

# Hanyu Pinyin for the highest-frequency Chinese surnames. unidecode already
# produces pinyin, but it renders each character separately with a trailing
# space and no tone-free surname convention; the explicit map keeps the surname
# as one token, which is what the blocking key depends on.
HAN_SURNAME_PINYIN: Final[dict[str, str]] = {
    "王": "wang", "李": "li", "张": "zhang", "刘": "liu", "陈": "chen",
    "杨": "yang", "黄": "huang", "赵": "zhao", "吴": "wu", "周": "zhou",
    "徐": "xu", "孙": "sun", "马": "ma", "朱": "zhu", "胡": "hu",
    "郭": "guo", "何": "he", "高": "gao", "林": "lin", "罗": "luo",
    "郑": "zheng", "梁": "liang", "谢": "xie", "宋": "song", "唐": "tang",
    "许": "xu", "韩": "han", "冯": "feng", "邓": "deng", "曹": "cao",
    "彭": "peng", "曾": "zeng", "肖": "xiao", "田": "tian", "董": "dong",
    "袁": "yuan", "潘": "pan", "于": "yu", "蒋": "jiang", "蔡": "cai",
    "余": "yu", "杜": "du", "叶": "ye", "程": "cheng", "苏": "su",
    "魏": "wei", "吕": "lv", "丁": "ding", "任": "ren", "沈": "shen",
    "姚": "yao", "卢": "lu", "姜": "jiang", "崔": "cui", "钟": "zhong",
    "谭": "tan", "陆": "lu", "汪": "wang", "范": "fan", "金": "jin",
    "石": "shi", "廖": "liao", "贾": "jia", "夏": "xia", "韦": "wei",
    "付": "fu", "方": "fang", "白": "bai", "邹": "zou", "孟": "meng",
    "熊": "xiong", "秦": "qin", "邱": "qiu", "江": "jiang", "尹": "yin",
    "薛": "xue", "闫": "yan", "段": "duan", "雷": "lei", "侯": "hou",
    "龙": "long", "史": "shi", "陶": "tao", "黎": "li", "贺": "he",
    "顾": "gu", "毛": "mao", "郝": "hao", "龚": "gong", "邵": "shao",
    "万": "wan", "钱": "qian", "严": "yan", "覃": "qin", "武": "wu",
    "戴": "dai", "莫": "mo", "孔": "kong", "向": "xiang", "汤": "tang",
    "习": "xi", "温": "wen", "岳": "yue", "章": "zhang",
}

_ARABIC_DEFINITE_ARTICLE: Final[str] = "ال"


def detect_script(text: str) -> Script:
    """Dominant script of `text`, ignoring digits, punctuation and whitespace.

    Dominant rather than exclusive: SDN entries routinely mix scripts, e.g.
    'ALI Hassan (علي حسن)'. Picking the majority script keeps the routing
    decision stable; genuinely mixed strings are handled token-wise downstream.
    """
    counts: dict[Script, int] = {}
    for char in text:
        if not char.isalpha():
            continue
        code = ord(char)
        script = Script.OTHER
        for low, high, candidate in _SCRIPT_RANGES:
            if low <= code <= high:
                script = candidate
                break
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return Script.LATIN
    return max(counts.items(), key=lambda item: (item[1], item[0].value))[0]


@dataclass(frozen=True)
class TransliterationResult:
    text: str
    source_script: Script
    method: Method
    lossy: bool

    @property
    def is_latin_source(self) -> bool:
        return self.source_script is Script.LATIN


def _transliterate_cyrillic(token: str) -> str:
    out: list[str] = []
    for char in token:
        lower = char.lower()
        mapped = CYRILLIC_MAP.get(lower)
        out.append(mapped if mapped is not None else (char if char.isascii() else ""))
    return "".join(out)


def _transliterate_arabic(token: str) -> tuple[str, Method]:
    known = ARABIC_NAME_LEXICON.get(token)
    if known is not None:
        return known, Method.LEXICON

    # The definite article fuses onto the following noun (الحربي = al-Harbi).
    # Stripping it before the lexicon lookup roughly doubles the hit rate, and
    # keeping "al" out of the output avoids a meaningless shared prefix that
    # would inflate Jaro-Winkler across every Arabic surname.
    if token.startswith(_ARABIC_DEFINITE_ARTICLE) and len(token) > 3:
        stem = token[len(_ARABIC_DEFINITE_ARTICLE):]
        known = ARABIC_NAME_LEXICON.get(stem)
        if known is not None:
            return known, Method.LEXICON
        return "".join(ARABIC_CHAR_MAP.get(c, "") for c in stem), Method.TABLE

    return "".join(ARABIC_CHAR_MAP.get(c, "") for c in token), Method.TABLE


def _transliterate_han(token: str) -> tuple[str, Method]:
    if not token:
        return "", Method.PINYIN
    parts: list[str] = []
    method = Method.PINYIN
    # Chinese personal names are surname-first and the surname is the leading
    # one (occasionally two) characters. Emitting surname and given name as
    # separate tokens is what lets the name-order-swap handling downstream work.
    surname = HAN_SURNAME_PINYIN.get(token[0])
    if surname is not None:
        parts.append(surname)
        remainder = token[1:]
    else:
        remainder = token
        method = Method.FALLBACK
    if remainder:
        given = unidecode(remainder).strip().lower().replace(" ", "")
        if given:
            parts.append(given)
        else:
            method = Method.FALLBACK
    return " ".join(p for p in parts if p), method


def _transliterate_token(token: str) -> tuple[str, Method]:
    script = detect_script(token)
    if script is Script.LATIN:
        return token, Method.IDENTITY
    if script is Script.CYRILLIC:
        return _transliterate_cyrillic(token), Method.TABLE
    if script is Script.ARABIC:
        return _transliterate_arabic(token)
    if script is Script.HAN:
        return _transliterate_han(token)
    return unidecode(token), Method.FALLBACK


def transliterate(text: str) -> TransliterationResult:
    """Reduce `text` to a Latin representation suitable for phonetic encoding.

    Raises TransliterationError when a non-empty input yields nothing usable.
    Returning the original string on failure would be worse: it would look like
    a successful pass and poison every downstream key.
    """
    if text is None:  # defensive: nulls are endemic in watchlist feeds
        raise TransliterationError("cannot transliterate None")
    stripped = text.strip()
    if not stripped:
        return TransliterationResult("", Script.LATIN, Method.IDENTITY, lossy=False)

    overall = detect_script(stripped)
    if overall is Script.LATIN and stripped.isascii():
        return TransliterationResult(stripped, Script.LATIN, Method.IDENTITY, lossy=False)

    tokens = stripped.split()
    pieces: list[str] = []
    methods: set[Method] = set()
    for token in tokens:
        converted, method = _transliterate_token(token)
        methods.add(method)
        if converted.strip():
            pieces.append(converted.strip())

    latin = " ".join(pieces)
    if overall is Script.LATIN:
        # Accented Latin (José, Müller) reaches here; NFKD + combining-mark
        # removal is the standard fold and is not considered lossy for matching.
        latin = _strip_combining_marks(latin)

    if not latin and stripped:
        latin = unidecode(stripped).strip()
        methods = {Method.FALLBACK}
    if not latin:
        raise TransliterationError(
            f"transliteration produced an empty Latin form for {text!r} (script={overall.value})"
        )

    if len(methods) == 1:
        method = next(iter(methods))
    else:
        method = Method.MIXED
    lossy = overall is not Script.LATIN and method in {Method.TABLE, Method.FALLBACK, Method.MIXED}
    return TransliterationResult(latin, overall, method, lossy=lossy)


def _strip_combining_marks(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def to_latin(text: str) -> str:
    return transliterate(text).text
