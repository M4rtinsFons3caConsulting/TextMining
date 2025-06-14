"""
Text Preprocessing Module for NLP Tasks

This module provides a comprehensive text preprocessing pipeline designed for
natural language processing applications, particularly in financial domains.

Features include:
- Emoji demojification (emoji → descriptive words)
- Ticker symbol handling (removal or anonymization)
- URL removal or anonymization
- Handle and hashtag removal
- Date, datetime, and time expression removal using dateparser
- Conversion of percentage change expressions (+5.5%) to tokens ('up'/'down')
- Contraction expansion (using contractions library)
- Removal of monetary prices
- Removal of punctuation, special characters, and ellipses
- Lemmatization and stemming with optional external tools
- Stopword removal (placed after stemming/lemmatization for accuracy)
- Removal of geographic locations (cities, countries, ISO codes) after stemming
- Case normalization support

Dependencies:
- re
- string
- emoji
- pycountry
- pandas
- contractions
- dateparser
- geotext
- typing
"""

import re
import string
from typing import Optional, Set

import emoji
import pycountry
import pandas as pd
import contractions
from dateparser.search import search_dates
from geotext import GeoText

# Prepare blacklist of ISO country codes (2-letter and 3-letter)
BLACK_LIST = list({
    country.alpha_2
    for country in pycountry.countries
}.union({
    country.alpha_3
    for country in pycountry.countries
}))

def preprocess(
    corpus: pd.Series,
    to_lower: Optional[bool] = False,
    keep_ticker: Optional[bool] = False,
    anonymize: Optional[bool] = False,
    keep_url: Optional[bool] = False,
    stopwords: Optional[Set[str]] = None,
    lemmatizer: Optional[object] = None,
    stemmizer: Optional[object] = None,
) -> pd.Series:
    """
    Apply a comprehensive preprocessing pipeline to a corpus of text data.

    Args:
        corpus (pd.Series): Series containing text documents.
        to_lower (Optional[bool]): Convert text to lowercase if True. Default False.
        keep_ticker (Optional[bool]): Keep stock tickers (e.g., $AAPL) if True. Default False.
        anonymize (Optional[bool]): Replace tickers with [TICKER] token if True, overrides keep_ticker.
        keep_url (Optional[bool]): Replace URLs with [URL] token if True, else remove URLs. Default False.
        stopwords (Optional[Set[str]]): Set of stopwords to remove after stemming/lemmatization.
        lemmatizer (Optional[object]): Object with .lemmatize(word) method for lemmatization.
        stemmizer (Optional[object]): Object with .stem(word) method for stemming.

    Returns:
        pd.Series: Preprocessed text documents as a pandas Series.
    """
    def _fix_acronyms(text: str) -> str:
        pattern = r"\b((?:[A-Z]\.){2,}[A-Z]?)(\.?)\b"
        
        def repl(m):
            acronym = m.group(1).replace(".", "")
            trailing_dot = m.group(2)
            return acronym + trailing_dot
        
        return re.sub(pattern, repl, text)

    def _delete_spaces(text: str) -> str:
        return re.sub(r"\s{2,}", " ", text).strip()

    def _demojify(text: str) -> str:
        """
        Replace emojis in the text with descriptive words.

        Example: "😀" → "grinning face"
        """
        demojized = emoji.demojize(text)
        return demojized.replace("_", " ").replace(":", "")

    def _clean_ticker(text: str) -> str:
        """
        Remove stock tickers (e.g. $AAPL) from the text.
        """
        return re.sub(r"\$[a-zA-Z]+", "", text)

    def _clean_url(text: str) -> str:
        """
        Remove all URLs starting with http(s).
        """
        return re.sub(r"http\S+", "", text)

    def _clean_prices(text: str) -> str:
        """
        Remove major currency symbols, ISO codes (e.g., USD, EUR), and combinations (e.g., C$, A$).
        """
        pattern = r"""
            \b(?:USD|EUR|CAD|GBP|JPY|CHF|AUD|NZD|CNY|HKD|SGD|INR)\b    
            |[€$£¥₹]                                                  
            |\b[A-Z]{1}\$                                             
            |\b[A-Z]\d+                                               
        """
        return re.sub(pattern, "", text, flags=re.IGNORECASE | re.VERBOSE)

    def _clean_punctuation(text: str) -> str:
        """
        Remove all punctuation characters and ellipses.
        """
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        return text.replace("…", "")

    def _clean_special_chars(text: str) -> str:
        """
        Remove or normalize miscellaneous special characters and symbols.
        """
        for char in ["’", "‘", '”', '“', "®", "™", "\x80", "→", "–", "•"]:
            text = text.replace(char, "")
        for char in ["\x8f", "\x9d", "—"]:
            text = text.replace(char, " ")
        return text

    def _remove_stopwords(text: str, stopwords: Set[str]) -> str:
        stopwords_upper = {w.upper() for w in stopwords} # Case insensitive
        return " ".join(word for word in text.split() if word.upper() not in stopwords_upper)

    def _lemmatize_text(text: str, lemmatizer: object) -> str:
        """
        Lemmatize each word, except for blacklisted ISO codes.
        """
        return " ".join(
            word if word in blacklist else lemmatizer.lemmatize(word)
            for word in text.split()
        )

    def _stemmize_text(text: str, stemmizer: object) -> str:
        """
        Stem each word, except for blacklisted ISO codes.
        """
        return " ".join(
            word if word in blacklist else stemmizer.stem(word)
            for word in text.split()
        )

    def _remove_handles(text: str) -> str:
        """
        Remove social media handles beginning with '@'.
        """
        return re.sub(r"@\w+", "", text)

    def _remove_hashtags(text: str) -> str:
        """
        Remove hashtags beginning with '#'.
        """
        return re.sub(r"#\w+", "", text)

    def _remove_dates_with_search(text: str) -> str:
        """
        Remove date, datetime, and time expressions detected by dateparser.
        """
        found = search_dates(text)
        if not found:
            return text
        for match_text, _ in found:
            escaped = re.escape(match_text)
            text = re.sub(rf'\b{escaped}\b', '', text)
        # Remove excess whitespace after removals
        return re.sub(r'\s{2,}', ' ', text).strip()

    def clean_remaining_date_time(text: str) -> str:
        days = r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"
        months = r"\b(0[1-9]|1[0-2]|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)\b"
        
        pattern = rf"{days}|{months}"
        
        return re.sub(pattern, "", text, flags=re.IGNORECASE)

    def _convert_percentage_changes(text: str) -> str:
        """
        Convert +X.X% or -X.X% expressions into 'up' or 'down' tokens.
        """
        def repl(match):
            sign = match.group(1)
            return "up" if sign == "+" else "down"
        return re.sub(r"([+-])\d+(\.\d+)?%", repl, text)

    def _remove_contractions(text: str) -> str:
        """
        Expand contractions (e.g., "can't" → "cannot") using contractions lib.
        """
        return contractions.fix(text)

    def _remove_possessives(text: str) -> str:
        """
        Remove possessive endings from words in the text.
        """
        text = re.sub(r"(\w+)'s\b", r"\1", text)   # Remove 's
        text = re.sub(r"(\w+)s'\b", r"\1s", text)  # Remove apostrophe after plural
        return text

    def _remove_locations(text: str) -> str:
        """
        Remove city names, country names, and ISO codes from text after stemming.

        Uses geotext for city and country detection.
        Also removes all ISO codes in blacklist (2 or 3 letters).
        """
        places = GeoText(text)
        # Remove detected cities (case-insensitive)
        for city in places.cities:
            text = re.sub(rf"\b{re.escape(city)}\b", "", text, flags=re.IGNORECASE)
        # Remove detected countries (case-insensitive)
        for country in places.countries:
            text = re.sub(rf"\b{re.escape(country)}\b", "", text, flags=re.IGNORECASE)
        # Remove ISO codes and optionally trailing punctuation
        for code in BLACK_LIST:
            pattern = rf"\b{r'\.?'.join(list(re.escape(code)))}\.?\b"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        # Clean extra spaces
        return re.sub(r'\s{2,}', ' ', text).strip()
    
    def remove_all_integers(text: str) -> str:
        return re.sub(r"\b\d+\b", "", text)
    
    # Standardize acronyms accross the corpus
    corpus = corpus.apply(_fix_acronyms).apply(_delete_spaces)

    # Convert emojies
    corpus = corpus.apply(_demojify)

    # Remove social media handles
    corpus = corpus.apply(_remove_handles).apply(_delete_spaces)

    # Renove hashtags
    corpus = corpus.apply(_remove_hashtags).apply(_delete_spaces)
    
    # Remove ticker symbols
    corpus = corpus.apply(_clean_ticker).apply(_delete_spaces)

    # Remove URL
    corpus = corpus.apply(_clean_url).apply(_delete_spaces)

    # Remove geographic locations and ISO codes
    corpus = corpus.apply(_remove_locations).apply(_delete_spaces)

    # Convert percentage changes to up/down tokens
    corpus = corpus.apply(_convert_percentage_changes)

    # Remove monetary prices
    corpus = corpus.apply(_clean_prices).apply(_delete_spaces)

    # Remove dates, times, and datetimes using dateparser
    corpus = corpus.apply(_remove_dates_with_search).apply(_delete_spaces)
    corpus = corpus.apply(clean_remaining_date_time).apply(_delete_spaces)

    # Expand contractions
    corpus = corpus.apply(_remove_contractions).apply(_delete_spaces)

    # Remove possessive markers 
    corpus = corpus.apply(_remove_possessives).apply(_delete_spaces)

    # Remove integers
    corpus = corpus.apply(remove_all_integers).apply(_delete_spaces)

    # Normalize case and prepare blacklist accordingly
    if to_lower:
        corpus = corpus.apply(str.lower)
        blacklist = {w.lower() for w in BLACK_LIST}
    else:
        blacklist = set(BLACK_LIST)

    # Remove punctuation and ellipses
    corpus = corpus.apply(_clean_punctuation).apply(_delete_spaces)

    # Remove miscellaneous special characters
    corpus = corpus.apply(_clean_special_chars).apply(_delete_spaces)

    # Lemmatize text if lemmatizer is provided
    if lemmatizer:
        corpus = corpus.apply(lambda x: _lemmatize_text(x, lemmatizer))

    # Stem text if stemmizer is provided
    if stemmizer:
        corpus = corpus.apply(lambda x: _stemmize_text(x, stemmizer))

    # Remove stopwords AFTER lemmatization/stemming
    if stopwords:
        corpus = corpus.apply(lambda x: _remove_stopwords(x, stopwords))

    # Remove geographic locations and ISO codes
    corpus = corpus.apply(_remove_locations).apply(_delete_spaces)

    return corpus