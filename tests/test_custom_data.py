import json
import zipfile
from pathlib import Path

import pytest

from inference_service.adapters.custom_artifacts import (
    CustomConfig,
    encode,
    read_manifest,
    sha256,
    tokens,
    vocabulary,
)
from training.data import DOMAINS, Row, prepare, read_split, split_rows
from training.metrics import classification_metrics


def test_vocabulary_fits_only_supplied_training_text() -> None:
    words = vocabulary(["GOOD good isn't bad", "fine"])
    assert words[:3] == ["<pad>", "<unk>", "good"]
    indices = {word: i for i, word in enumerate(words)}
    assert encode("unseen validationword", indices, 256) == [1, 1]
    assert encode("!!!", indices, 256) == [1]
    assert encode("good " * 300, indices, 256) == [2] * 256
    assert tokens("ISN'T bad!") == ["isn't", "bad"]


def test_split_deduplicates_before_stratified_partition() -> None:
    rows = [
        Row(
            id=str(i),
            text=f"word {chr(97 + i // 26)}{chr(97 + i % 26)}",
            label=i % 2,
            domain="fixture",
        )
        for i in range(100)
    ]
    rows += [Row(id="duplicate", text=rows[0]["text"].upper(), label=0, domain="other")]
    rows += [Row(id="conflict", text=rows[1]["text"], label=0, domain="other")]
    splits, removed = split_rows(rows, 42)
    assert (splits, removed) == split_rows(rows, 42)
    assert removed == 3
    canonical = [{" ".join(tokens(r["text"])) for r in split} for split in splits.values()]
    assert all(canonical)
    assert not canonical[0] & canonical[1]
    assert not canonical[0] & canonical[2]
    assert not canonical[1] & canonical[2]
    assert sum(map(len, canonical)) == 99


def test_metrics_include_both_classes_for_trivial_baseline() -> None:
    metrics = classification_metrics([0, 0, 1, 1], [0, 0, 0, 0])
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert metrics["confusion_matrix_true_by_predicted"] == [[2, 0], [2, 0]]
    assert classification_metrics([0, 1], [0, 1])["macro_f1"] == 1.0
    with pytest.raises(ValueError):
        classification_metrics([], [])


def test_missing_artifacts_fail_clearly(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="manifest"):
        read_manifest(tmp_path)


def test_incompatible_model_config_rejected() -> None:
    with pytest.raises(ValueError):
        CustomConfig.model_validate({"labels": ["positive", "negative"]})
    with pytest.raises(ValueError):
        CustomConfig(max_length=0)


def test_prepare_preserves_unicode_sentence_separators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for domain in DOMAINS:
            lines = [
                f"{domain} word {chr(97 + i // 676)}{chr(97 + i // 26 % 26)}"
                f"{chr(97 + i % 26)}\u2028sentence\t{i % 2}\n"
                for i in range(1000)
            ]
            bundle.writestr(f"sentiment labelled sentences/{domain}_labelled.txt", "".join(lines))
    monkeypatch.setattr("training.data.ARCHIVE_SHA256", sha256(archive))
    output = tmp_path / "prepared"
    prepare(archive, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["raw_count"] == 3000
    assert sum(manifest["splits"][s]["count"] for s in ("train", "validation", "test")) == 3000
    assert all("\u2028" in r["text"] for r in read_split(output, "train"))
    with pytest.raises(FileExistsError):
        prepare(archive, output)


def test_changed_split_fails_integrity_check(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"splits": {"train": {"sha256": "0" * 64}}}))
    (tmp_path / "train.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="integrity"):
        read_split(tmp_path, "train")
