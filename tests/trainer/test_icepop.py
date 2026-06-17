"""Unit tests for IcePop masked mismatch correction (token + sequence level)."""

import math

import pytest

torch = pytest.importorskip("torch")
tinker = pytest.importorskip("tinker")

from omegaconf import OmegaConf  # noqa: E402

from rllm.trainer.algorithms.config import AlgorithmConfig, RolloutCorrectionConfig  # noqa: E402
from rllm.trainer.fireworks.fireworks_policy_trainer import FireworksPolicyTrainer  # noqa: E402

compute_weight = FireworksPolicyTrainer._compute_icepop_weight
build_datums = FireworksPolicyTrainer._build_icepop_builtin_loss_datums


class TestComputeIcepopWeight:
    def test_token_mode_masks_outside_band(self):
        # ratios: e^0 = 1 (keep), e^1 ~ 2.72 (> high, drop), e^-1 ~ 0.37 (< low, drop)
        prox = torch.tensor([0.0, -1.0, -3.0])
        inf = torch.tensor([0.0, -2.0, -2.0])
        weight, rho, keep = compute_weight(prox, inf, "token", 0.5, 2.0)

        assert rho.tolist() == pytest.approx([1.0, math.e, math.exp(-1.0)], rel=1e-5)
        assert keep.tolist() == [True, False, False]
        assert weight.tolist() == pytest.approx([1.0, 0.0, 0.0])

    def test_token_mode_keeps_rho_inside_band(self):
        prox = torch.tensor([-1.0, -1.5])
        inf = torch.tensor([-1.2, -1.3])
        weight, rho, keep = compute_weight(prox, inf, "token", 0.5, 2.0)
        assert keep.all()
        assert weight.tolist() == pytest.approx(rho.tolist())

    def test_sequence_mode_uses_geometric_mean(self):
        # Per-token log ratios: +1 and -1 -> mean 0 -> seq rho = 1 everywhere,
        # even though each token alone would fall outside a tight band.
        prox = torch.tensor([-1.0, -3.0])
        inf = torch.tensor([-2.0, -2.0])
        weight, rho, keep = compute_weight(prox, inf, "sequence", 0.9, 1.1)

        assert rho.tolist() == pytest.approx([1.0, 1.0])
        assert keep.all()
        assert weight.tolist() == pytest.approx([1.0, 1.0])

    def test_sequence_mode_drops_whole_sequence(self):
        prox = torch.tensor([-1.0, -1.0])
        inf = torch.tensor([-2.0, -2.0])  # seq rho = e > high
        weight, _, keep = compute_weight(prox, inf, "sequence", 0.5, 2.0)
        assert not keep.any()
        assert weight.tolist() == pytest.approx([0.0, 0.0])

    def test_custom_asymmetric_band(self):
        prox = torch.tensor([0.0, 0.0])
        inf = torch.tensor([math.log(0.6), math.log(2.5)])  # rhos ~ 1.67, 0.4
        # Wide on the high side, tight on the low side.
        weight, rho, keep = compute_weight(prox, inf, "token", 0.5, 3.0)
        assert keep.tolist() == [True, False]
        assert weight.tolist() == pytest.approx([rho[0].item(), 0.0])

    def test_log_ratio_clamped(self):
        prox = torch.tensor([0.0])
        inf = torch.tensor([-100.0])
        _, rho, _ = compute_weight(prox, inf, "token", 0.5, 2.0)
        assert rho.item() == pytest.approx(math.exp(20.0))

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="icepop_mode"):
            compute_weight(torch.tensor([0.0]), torch.tensor([0.0]), "both", 0.5, 2.0)

    def test_invalid_band_raises(self):
        with pytest.raises(ValueError, match="0 < low < high"):
            compute_weight(torch.tensor([0.0]), torch.tensor([0.0]), "token", 2.0, 0.5)


class TestIcepopConfig:
    def test_default_band_is_symmetric(self):
        rc = RolloutCorrectionConfig(icepop_mode="token", icepop_beta=2.0)
        assert rc.icepop_bounds == (0.5, 2.0)

    def test_custom_alpha_overrides_lower_bound(self):
        rc = RolloutCorrectionConfig(icepop_mode="sequence", icepop_alpha=0.8, icepop_beta=3.0)
        assert rc.icepop_bounds == (0.8, 3.0)

    def test_from_config_parses_icepop_keys(self):
        cfg = OmegaConf.create(
            {
                "adv_estimator": "grpo",
                "rollout_correction": {"icepop_mode": "sequence", "icepop_alpha": 0.7, "icepop_beta": 1.5, "bypass_mode": False},
            }
        )
        algo = AlgorithmConfig.from_config(cfg)
        assert algo.rollout_correction.icepop_mode == "sequence"
        assert algo.rollout_correction.icepop_bounds == (0.7, 1.5)
        assert algo.rollout_correction.bypass_mode is False


def make_clean_datum(prompt_len: int, resp_len: int) -> tinker.Datum:
    """Clean datum in the cookbook layout: full token sequence + loss mask."""
    n = prompt_len - 1 + resp_len  # right-shifted targets
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(range(n))),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=list(range(1, n + 1)), dtype="int64", shape=[n]),
            "loss_mask": tinker.TensorData(data=[0.0] * (prompt_len - 1) + [1.0] * resp_len, dtype="float32", shape=[n]),
        },
    )


class TestBuildIcepopDatums:
    def test_advantages_folded_with_weight_and_mask(self):
        prompt_len, resp_len = 3, 4
        datum = make_clean_datum(prompt_len, resp_len)
        n = prompt_len - 1 + resp_len
        # Response tokens: rho = [1, 1, e, 1] -> third token outside [0.5, 2.0].
        prox = [0.0] * n
        inf = [0.0] * (prompt_len - 1) + [0.0, 0.0, -1.0, 0.0]

        datums, metrics = build_datums(
            [datum],
            advantages=[2.0],
            prox_logprobs=[prox],
            inf_logprobs=[inf],
            prompt_lens=[prompt_len],
            icepop_mode="token",
            icepop_low=0.5,
            icepop_high=2.0,
        )

        assert len(datums) == 1
        adv = list(datums[0].loss_fn_inputs["advantages"].data)
        # Prompt positions zero; response: adv * weight * mask.
        assert adv[: prompt_len - 1] == [0.0, 0.0]
        assert adv[prompt_len - 1 :] == pytest.approx([2.0, 2.0, 0.0, 2.0])
        # Kernel logprobs are the proximal logprobs.
        assert list(datums[0].loss_fn_inputs["logprobs"].data) == pytest.approx(prox)

        assert metrics["rollout_correction/icepop/active_tokens"] == resp_len
        assert metrics["rollout_correction/icepop/zero_frac"] == pytest.approx(0.25)
        assert metrics["rollout_correction/icepop/high_frac"] == pytest.approx(0.25)
        assert metrics["rollout_correction/icepop/low_frac"] == 0.0

    def test_sequence_mode_emits_seq_metrics_and_masks_all(self):
        prompt_len, resp_len = 2, 3
        datum = make_clean_datum(prompt_len, resp_len)
        n = prompt_len - 1 + resp_len
        prox = [0.0] * n
        inf = [0.0] * (prompt_len - 1) + [-1.0] * resp_len  # seq rho = e > 2.0

        datums, metrics = build_datums(
            [datum],
            advantages=[1.0],
            prox_logprobs=[prox],
            inf_logprobs=[inf],
            prompt_lens=[prompt_len],
            icepop_mode="sequence",
            icepop_low=0.5,
            icepop_high=2.0,
        )

        adv = list(datums[0].loss_fn_inputs["advantages"].data)
        assert adv[prompt_len - 1 :] == pytest.approx([0.0, 0.0, 0.0])
        assert metrics["rollout_correction/icepop/zero_frac"] == pytest.approx(1.0)
        assert metrics["rollout_correction/icepop/seq_ratio/mean"] == pytest.approx(math.e, rel=1e-5)


def make_backend(rollout_correction: dict):
    from rllm.trainer.fireworks.fireworks_backend import FireworksBackend

    config = OmegaConf.create(
        {
            "training": {},
            "fuse_forward_backward_and_optim_step": False,
            "rllm": {
                "rollout": {"train": {}, "val": {}},
                "algorithm": {
                    "loss_fn": None,
                    "loss_agg_mode": None,
                    "eps_clip_high": None,
                    "router_replay": "disabled",
                    "rollout_correction": rollout_correction,
                },
                "trainer": {"save_freq": -1},
                "async_training": {"enable": False},
            },
        }
    )
    return FireworksBackend(config)


class TestBackendValidation:
    def test_valid_token_and_sequence_modes(self):
        for mode in ("token", "sequence"):
            make_backend({"icepop_mode": mode, "icepop_beta": 2.0, "bypass_mode": False}).validate_config()

    def test_custom_band_accepted(self):
        make_backend({"icepop_mode": "token", "icepop_alpha": 0.6, "icepop_beta": 1.8, "bypass_mode": False}).validate_config()

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="icepop_mode"):
            make_backend({"icepop_mode": "both", "bypass_mode": False}).validate_config()

    def test_beta_must_exceed_one(self):
        with pytest.raises(ValueError, match="icepop_beta"):
            make_backend({"icepop_mode": "token", "icepop_beta": 0.9, "bypass_mode": False}).validate_config()

    def test_alpha_must_be_in_unit_interval(self):
        with pytest.raises(ValueError, match="icepop_alpha"):
            make_backend({"icepop_mode": "token", "icepop_alpha": 1.5, "bypass_mode": False}).validate_config()

    def test_mutually_exclusive_with_tis(self):
        with pytest.raises(ValueError, match="only one"):
            make_backend({"icepop_mode": "token", "tis_mode": "token", "bypass_mode": False}).validate_config()
