import copy

import pytest

from budget_sid.constants import GRID_COMMIT, TEXT_ENCODER
from budget_sid.errors import ConfigRejected


def test_beauty_config_contract(beauty_config):
    assert beauty_config["text_encoder"]["name"] == TEXT_ENCODER
    assert beauty_config["text_encoder"]["embedding_dim"] == 384
    assert beauty_config["text_encoder"]["frozen"] is True
    assert beauty_config["semantic_id"]["num_codebooks"] == 3
    assert beauty_config["semantic_id"]["dedup_digit"] is True
    assert beauty_config["semantic_id"]["num_hierarchies"] == 4
    assert beauty_config["model"]["d_model"] == 128
    assert beauty_config["model"]["num_encoder_layers"] == 4
    assert beauty_config["model"]["num_decoder_layers"] == 4
    assert beauty_config["model"]["num_heads"] == 4
    assert beauty_config["train"]["batch_size"] == 16
    assert beauty_config["train"]["grad_accum"] == 4
    assert beauty_config["train"]["devices"] == 1
    assert beauty_config["train"]["strategy"] == "single_device"
    assert beauty_config["inference"]["beam_widths"] == [20, 40, 60]
    assert beauty_config["inference"]["top_k"] == 20
    assert beauty_config["grid"]["commit"] == GRID_COMMIT
    assert beauty_config["grid"]["vendor"] is False
    assert beauty_config["eval"]["sampled_negatives"] is False


def _reject(mutator, beauty_config):
    from budget_sid.config import validate_config

    data = copy.deepcopy(beauty_config)
    mutator(data)
    with pytest.raises(ConfigRejected):
        validate_config(data)


def test_reject_flan_t5_xl_backbone(beauty_config):
    _reject(lambda data: data["model"].__setitem__("backbone", "google/flan-t5-xl"), beauty_config)


def test_reject_flan_t5_xl_text_encoder(beauty_config):
    _reject(lambda data: data["text_encoder"].__setitem__("name", "google/FLAN-T5-XL"), beauty_config)


def test_reject_ddp_strategy(beauty_config):
    _reject(lambda data: data["train"].__setitem__("strategy", "ddp"), beauty_config)
    _reject(
        lambda data: data.__setitem__("trainer", {"strategy": "ddp_find_unused_parameters_true"}),
        beauty_config,
    )


def test_reject_devices_all_gpus(beauty_config):
    from budget_sid.config import validate_config

    data = copy.deepcopy(beauty_config)
    data["train"]["devices"] = -1
    with pytest.raises(ConfigRejected):
        validate_config(data)


def test_reject_grid_embedding_default():
    from budget_sid.config import validate_config

    with pytest.raises(ConfigRejected):
        validate_config({"embedding_model": "google/flan-t5-xl", "trainer": {"strategy": "ddp", "devices": -1}})
