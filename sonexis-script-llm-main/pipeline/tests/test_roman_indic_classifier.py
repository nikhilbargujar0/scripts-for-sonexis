"""Tests for roman_indic_classifier."""
import os

import pytest

from pipeline.roman_indic_classifier import RomanIndicClassifier


@pytest.fixture(scope="module")
def classifier(tmp_path_factory):
    cache = tmp_path_factory.mktemp("classifier") / "model.pkl"
    clf = RomanIndicClassifier(cache_path=str(cache))
    clf.train(persist=True)
    assert cache.exists()
    return clf


def test_classifier_loads_from_cache(tmp_path):
    cache = tmp_path / "model.pkl"
    a = RomanIndicClassifier(cache_path=str(cache))
    a.train(persist=True)
    assert cache.exists()
    b = RomanIndicClassifier(cache_path=str(cache))
    assert b.available() is True
    pred = b.predict("hello how are you doing today my friend")
    assert pred is not None


def test_english_prediction(classifier):
    pred = classifier.predict(
        "could you please share the invoice and confirmation number"
    )
    assert pred.language == "en"
    assert pred.confidence >= 0.5


def test_hinglish_prediction(classifier):
    pred = classifier.predict(
        "haan bhai main theek hoon tum batao kya chal raha hai"
    )
    assert pred.language == "hi-Latn"
    assert pred.confidence >= 0.5


def test_punjabi_prediction(classifier):
    pred = classifier.predict(
        "sat sri akaal paaji tusi kida ho asi vadhia haan"
    )
    assert pred.language == "pa-Latn"
    assert pred.confidence >= 0.5


def test_empty_input_returns_none(classifier):
    assert classifier.predict("") is None
    assert classifier.predict("   ") is None


def test_probabilities_sum_to_one(classifier):
    pred = classifier.predict("matlab mujhe samajh nahi aa raha yaar")
    total = sum(pred.probabilities.values())
    assert abs(total - 1.0) < 1e-6


def test_classifier_integrated_with_detect_language(classifier):
    from pipeline.language_detection import detect_language

    r = detect_language(
        "haan yaar main theek hoon aaj bahut kaam tha office mein",
        segments_text=["haan yaar main theek hoon aaj bahut kaam tha office mein"],
        roman_indic_classifier=classifier,
    )
    assert r.primary_language == "hi-Latn"
    assert "ml" in r.method
