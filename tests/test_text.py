"""Tests for tokenization and the message-length transformer."""

from disaster_response.text import TextLengthExtractor, tokenize


def test_tokenize_replaces_urls():
    tokens = tokenize("See http://example.com/flood for details")
    assert "urlplaceholder" in tokens
    assert not any("example" in token for token in tokens)


def test_tokenize_removes_stopwords_and_lowercases():
    tokens = tokenize("The Flood Is Coming To Our Village")
    assert tokens == ["flood", "come", "village"]


def test_tokenize_lemmatizes_verbs():
    assert "run" in tokenize("people are running for shelter")
    assert "need" in tokenize("families needed clean water")


def test_tokenize_strips_punctuation():
    assert tokenize("Help!!! Water... needed???") == ["help", "water", "need"]


def test_tokenize_empty_input():
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize(None) == []


def test_text_length_extractor_shape_and_values():
    extractor = TextLengthExtractor()
    result = extractor.fit_transform(["one two three", "hello"])
    assert result.shape == (2, 1)
    assert result[:, 0].tolist() == [3, 1]
