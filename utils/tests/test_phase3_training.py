import os
import pathlib

import pytest
from sentence_transformers import SentenceTransformer

from phase3.train_sentence_transformer import (
    apply_layer_freezing,
    create_discriminative_param_groups,
    generate_training_plots,
)

TMP_DIR = pathlib.Path(__file__).resolve().parent / "__tmp__"
TMP_DIR.mkdir(exist_ok=True)
os.environ.setdefault("TMPDIR", str(TMP_DIR))
os.environ.setdefault("TEMP", str(TMP_DIR))
os.environ.setdefault("TMP", str(TMP_DIR))


@pytest.fixture(scope="module")
def biobert_model():
    model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    yield model


def test_apply_layer_freezing_disables_lower_layers(biobert_model):
    transformer = biobert_model[0]
    status = apply_layer_freezing(transformer, freeze_layers=6)
    assert status["embeddings"] is False
    for idx in range(12):
        trainable = status[f"encoder.layer.{idx}"]
        if idx < 6:
            assert trainable is False
        else:
            assert trainable is True


def test_discriminative_param_groups_order(biobert_model):
    apply_layer_freezing(biobert_model[0], freeze_layers=6)
    param_groups, summary = create_discriminative_param_groups(
        biobert_model,
        freeze_layers=6,
        min_lr=1e-6,
        max_lr=5e-5,
        weight_decay=0.01,
    )
    assert param_groups
    lrs = [group["lr"] for group in param_groups]
    assert min(lrs) >= 1e-6
    assert max(lrs) <= 5e-5
    assert lrs == sorted(lrs), "Learning rates should be non-decreasing across groups"
    # Ensure summary is aligned with param_groups
    assert len(summary) == len(param_groups)


def test_generate_training_plots_creates_files(tmp_path):
    history = [
        {
            "epoch": 1,
            "train_loss": 1.2,
            "train_pearson": 0.10,
            "train_spearman": 0.12,
            "val_pearson": 0.15,
            "val_spearman": 0.13,
        },
        {
            "epoch": 2,
            "train_loss": 0.9,
            "train_pearson": 0.20,
            "train_spearman": 0.22,
            "val_pearson": 0.25,
            "val_spearman": 0.23,
        },
    ]
    paths = generate_training_plots(history, tmp_path)
    assert "loss_curve" in paths
    assert "correlation_curve" in paths
    for plot_path in paths.values():
        assert pathlib.Path(plot_path).exists()
