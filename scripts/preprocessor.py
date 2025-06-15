import re
import string
from typing import Optional, Set, Dict, Any
import pandas as pd
import emoji
import pycountry
import contractions
from dateparser.search import search_dates
from geotext import GeoText

# Prepare blacklist of ISO country codes (2-letter and 3-letter)
BLACK_LIST = list({
    country.alpha_2 for country in pycountry.countries
}.union({
    country.alpha_3 for country in pycountry.countries
}))

# Default preprocessing configuration
DEFAULT_CFG = {
    "fix_acronyms": True,
    "delete_spaces": True,
    "demojify": True,
    "clean_ticker": True,
    "keep_ticker": False,
    "anonymize_ticker": False,
    "clean_url": True,
    "keep_url": False,
    "clean_handles": False,
    "keep_handle": False,
    "clean_hashtags": True,
    "keep_hashtag": False,
    "clean_prices": True,
    "remove_punctuation": True,
    "remove_special_chars": True,
    "remove_stopwords": True,
    "lemmatize_text": True,
    "stem_text": False,
    "remove_dates_with_search": True,
    "clean_remaining_date_time": True,
    "convert_percentage_changes": True,
    "remove_contractions": True,
    "remove_possessives": True,
    "remove_locations": True,
    "remove_all_integers": True,
    "to_lower": True
}

def preprocess(
    corpus: pd.Series,
    cfg: Optional[Dict[str, bool]] = {},
    stopwords: Optional[Set[str]] = None,
    lemmatizer: Optional[Any] = None,
    stemmizer: Optional[Any] = None,
) -> pd.Series:
    """
    Apply a comprehensive text preprocessing pipeline to a corpus of text.

    Args:
        corpus (pd.Series): Series of text documents to preprocess.
        cfg (Dict[str, Any]): Configuration dictionary controlling each preprocessing step.
            Expected boolean keys include:
                - 'fix_acronyms'
                - 'delete_spaces'
                - 'demojify'
                - 'clean_ticker'
                - 'keep_ticker'
                - 'anonymize_ticker'
                - 'clean_url'
                - 'keep_url'
                - 'clean_handles'
                - 'keep_handle'
                - 'clean_hashtags'
                - 'keep_hashtag'
                - 'clean_prices'
                - 'remove_punctuation'
                - 'remove_special_chars'
                - 'remove_stopwords'
                - 'lemmatize_text'
                - 'stem_text'
                - 'remove_dates_with_search'
                - 'clean_remaining_date_time'
                - 'convert_percentage_changes'
                - 'remove_contractions'
                - 'remove_possessives'
                - 'remove_locations'
                - 'remove_all_integers'
                - 'to_lower'
        stopwords (Optional[Set[str]]): Set of stopwords to remove if configured.
        lemmatizer (Optional[Any]): Lemmatizer object with `.lemmatize(word)` method.
        stemmizer (Optional[Any]): Stemmer object with `.stem(word)` method.

    Returns:
        pd.Series: The processed text corpus.
    """
    def _fix_acronyms(text: str) -> str:
        """Remove dots from acronyms, e.g. U.S.A. -> USA"""
        pattern = r"\b((?:[A-Z]\.){2,}[A-Z]?)(\.?)\b"
        def repl(m):
            acronym = m.group(1).replace(".", "")
            trailing_dot = m.group(2)
            return acronym + trailing_dot
        return re.sub(pattern, repl, text)

    def _delete_spaces(text: str) -> str:
        """Collapse multiple spaces into one and strip leading/trailing spaces."""
        return re.sub(r"\s{2,}", " ", text).strip()

    def _demojify(text: str) -> str:
        """Convert emojis into descriptive text (e.g., 😀 → grinning face)."""
        demojized = emoji.demojize(text)
        return demojized.replace("_", " ").replace(":", "")

    def _clean_ticker(text: str, keep_ticker: bool, anonymize_ticker: bool) -> str:
        """
        Process stock tickers.
        Replace with [TICKER] if anonymize_ticker=True,
        else remove if keep_ticker=False,
        else keep unchanged.
        """
        if anonymize_ticker:
            return re.sub(r"\$[a-zA-Z]+", "[TICKER]", text)
        elif keep_ticker:
            return text
        else:
            return re.sub(r"\$[a-zA-Z]+", "", text)

    def _clean_url(text: str, keep_url: bool) -> str:
        """Replace URLs with [URL] if keep_url else remove URLs."""
        return re.sub(r"http\S+", "[URL]" if keep_url else "", text)

    def _clean_handles(text: str, keep_handle: bool) -> str:
        """Replace or remove social media handles starting with '@'."""
        return re.sub(r"@\w+", "[HANDLE]" if keep_handle else "", text)

    def _clean_hashtags(text: str, keep_hashtag: bool) -> str:
        """Replace or remove hashtags starting with '#'."""
        return re.sub(r"#\w+", "[HASHTAG]" if keep_hashtag else "", text)

    def _clean_prices(text: str) -> str:
        """Remove currency symbols and ISO currency codes."""
        pattern = r"""
            \b(?:USD|EUR|CAD|GBP|JPY|CHF|AUD|NZD|CNY|HKD|SGD|INR)\b
            |[€$£¥₹]
            |\b[A-Z]{1}\$
        """
        return re.sub(pattern, "", text, flags=re.IGNORECASE | re.VERBOSE)

    def _remove_punctuation(text: str) -> str:
        """Remove punctuation characters and ellipses."""
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        return text.replace("…", "")

    def _remove_special_chars(text: str) -> str:
        """Remove miscellaneous special characters."""
        for char in ["’", "‘", '”', '“', "®", "™", "\x80", "→", "–", "•"]:
            text = text.replace(char, "")
        for char in ["\x8f", "\x9d", "—"]:
            text = text.replace(char, " ")
        return text

    def _remove_stopwords(text: str, stopwords: Set[str]) -> str:
        """Remove stopwords from text."""
        stopwords_upper = {w.upper() for w in stopwords}
        return " ".join(word for word in text.split() if word.upper() not in stopwords_upper)

    def _lemmatize_text(text: str, lemmatizer: Any) -> str:
        """Lemmatize each word except blacklisted ISO codes."""
        blacklist = set(BLACK_LIST)
        return " ".join(
            word if word in blacklist else lemmatizer.lemmatize(word)
            for word in text.split()
        )

    def _stemmize_text(text: str, stemmizer: Any) -> str:
        """Stem each word except blacklisted ISO codes."""
        blacklist = set(BLACK_LIST)
        return " ".join(
            word if word in blacklist else stemmizer.stem(word)
            for word in text.split()
        )

    def _remove_dates_with_search(text: str) -> str:
        """Remove date/time expressions detected by dateparser."""
        found = search_dates(text)
        if not found:
            return text
        for match_text, _ in found:
            escaped = re.escape(match_text)
            text = re.sub(rf'\b{escaped}\b', '', text)
        return re.sub(r'\s{2,}', ' ', text).strip()

    def _clean_remaining_date_time(text: str) -> str:
        """Remove leftover day/month names."""
        days = r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"
        months = r"\b(0[1-9]|1[0-2]|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)\b"
        pattern = rf"{days}|{months}"
        return re.sub(pattern, "", text, flags=re.IGNORECASE)

    def _convert_percentage_changes(text: str) -> str:
        """Convert +X.X% or -X.X% to 'up' or 'down' tokens."""
        def repl(match):
            return "up" if match.group(1) == "+" else "down"
        return re.sub(r"([+-])\d+(\.\d+)?%", repl, text)

    def _remove_contractions(text: str) -> str:
        """Expand contractions using the contractions library."""
        return contractions.fix(text)

    def _remove_possessives(text: str) -> str:
        """Remove possessive endings from words."""
        text = re.sub(r"(\w+)'s\b", r"\1", text)
        text = re.sub(r"(\w+)s'\b", r"\1s", text)
        return text

    def _remove_locations(text: str) -> str:
        """Remove cities, countries, and ISO codes from text."""
        places = GeoText(text)
        for city in places.cities:
            text = re.sub(rf"\b{re.escape(city)}\b", "", text, flags=re.IGNORECASE)
        for country in places.countries:
            text = re.sub(rf"\b{re.escape(country)}\b", "", text, flags=re.IGNORECASE)
        for code in BLACK_LIST:
            pattern = rf"\b{r'\.?'.join(list(re.escape(code)))}\.?\b"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return re.sub(r'\s{2,}', ' ', text).strip()

    def _remove_all_integers(text: str) -> str:
        """Remove all standalone integers."""
        return re.sub(r"\b\d+\b", "", text)

    # Override defaults with user config values
    cfg = {**DEFAULT_CFG, **cfg}

    # Build dynamic mapping
    mapping = {
        "fix_acronyms": _fix_acronyms,
        "delete_spaces": _delete_spaces,
        "demojify": _demojify,
        "clean_ticker": lambda text: _clean_ticker(text, cfg.get("keep_ticker", False), cfg.get("anonymize_ticker", False)),
        "clean_url": lambda text: _clean_url(text, cfg.get("keep_url", False)),
        "clean_handles": lambda text: _clean_handles(text, cfg.get("keep_handle", False)),
        "clean_hashtags": lambda text: _clean_hashtags(text, cfg.get("keep_hashtag", False)),
        "clean_prices": _clean_prices,
        "remove_punctuation": _remove_punctuation,
        "remove_special_chars": _remove_special_chars,
        "remove_stopwords": lambda text: _remove_stopwords(text, stopwords) if stopwords else text,
        "lemmatize_text": lambda text: _lemmatize_text(text, lemmatizer) if lemmatizer else text,
        "stem_text": lambda text: _stemmize_text(text, stemmizer) if stemmizer else text,
        "remove_dates_with_search": _remove_dates_with_search,
        "clean_remaining_date_time": _clean_remaining_date_time,
        "convert_percentage_changes": _convert_percentage_changes,
        "remove_contractions": _remove_contractions,
        "remove_possessives": _remove_possessives,
        "remove_locations": _remove_locations,
        "remove_all_integers": _remove_all_integers,
        "to_lower": lambda text: text.lower(),
    }
    
    # Execute steps according to cfg and update corpus sequentially:
    for key, func in mapping.items():
        if cfg.get(key, False):
            corpus = corpus.apply(func).apply(_delete_spaces)

    return corpus
   