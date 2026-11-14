"""Text processing: tokenization and custom sklearn transformers.

NLTK resources are downloaded on demand (see :func:`ensure_nltk_resources`)
instead of at import time, so importing this module never hits the network.
"""

from __future__ import annotations

import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

#: NLTK resources the tokenizer needs. ``punkt_tab`` is required by
#: :func:`nltk.tokenize.word_tokenize` on NLTK >= 3.9.
NLTK_RESOURCES = ("punkt", "punkt_tab", "wordnet", "stopwords")

_URL_REGEX = (
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]"
    r"|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)

_lemmatizer: WordNetLemmatizer | None = None
_stopwords: frozenset[str] | None = None


def ensure_nltk_resources(download_dir: str | None = None) -> None:
    """Download any missing NLTK resources (no-op when all are present).

    Parameters
    ----------
    download_dir:
        Optional directory to download into. Defaults to NLTK's standard
        search path (``$NLTK_DATA`` or ``~/nltk_data``).
    """
    for resource in NLTK_RESOURCES:
        try:
            nltk.data.find(
                f"{'tokenizers' if resource.startswith('punkt') else 'corpora'}"
                f"/{resource}"
            )
        except LookupError:
            nltk.download(resource, download_dir=download_dir, quiet=True)


def _get_lemmatizer() -> WordNetLemmatizer:
    global _lemmatizer
    if _lemmatizer is None:
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


def _get_stopwords() -> frozenset[str]:
    global _stopwords
    if _stopwords is None:
        _stopwords = frozenset(stopwords.words("english"))
    return _stopwords


def tokenize(text: str) -> list[str]:
    """Normalize and tokenize a disaster message.

    URLs are replaced with a placeholder, punctuation is stripped,
    English stopwords are removed, and remaining tokens are lowercased
    and lemmatized as verbs (e.g. ``"running"`` -> ``"run"``).

    Parameters
    ----------
    text:
        Raw message string. Empty or non-string input yields ``[]``.

    Returns
    -------
    list[str]
        Cleaned tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    text = re.sub(_URL_REGEX, "urlplaceholder", text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    tokens = [
        token for token in word_tokenize(text)
        if token not in _get_stopwords()
    ]
    lemmatizer = _get_lemmatizer()
    return [lemmatizer.lemmatize(token, pos="v").strip() for token in tokens]


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """sklearn transformer returning each message's token count.

    Used as a second feature branch alongside TF-IDF: longer messages
    carry signal about the message type (e.g. direct reports vs. news).
    """

    def fit(self, X, y=None):  # noqa: N803 - sklearn convention
        return self

    def transform(self, X):
        lengths = pd.Series(X).apply(lambda text: len(word_tokenize(str(text))))
        return lengths.to_numpy().reshape(-1, 1)


if __name__ == "__main__":
    ensure_nltk_resources()
    print("NLTK resources ready:", ", ".join(NLTK_RESOURCES))
