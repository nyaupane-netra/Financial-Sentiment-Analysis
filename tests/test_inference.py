from types import SimpleNamespace

import pytest

from inference import probabilities_by_label, require_local_model, resolve_label_map


def model_with_labels(labels):
    return SimpleNamespace(config=SimpleNamespace(id2label=labels))


def test_resolve_label_map_preserves_model_class_order():
    model = model_with_labels({0: "positive", 1: "negative", 2: "neutral"})
    assert resolve_label_map(model) == {
        0: "positive",
        1: "negative",
        2: "neutral",
    }


def test_resolve_label_map_supports_generic_custom_checkpoint():
    model = model_with_labels({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
    assert resolve_label_map(model) == {0: "negative", 1: "neutral", 2: "positive"}


def test_probabilities_are_returned_in_canonical_order():
    result = probabilities_by_label(
        [0.8, 0.1, 0.1], {0: "positive", 1: "negative", 2: "neutral"}
    )
    assert list(result) == ["negative", "neutral", "positive"]
    assert result["positive"] == pytest.approx(0.8)


def test_require_local_model_rejects_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        require_local_model(tmp_path / "missing")
