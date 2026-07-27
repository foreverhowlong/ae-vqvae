from training.run_experiment_sequence import make_run_name


def test_ablation_label_owns_run_name():
    parameters = {
        "ablation": "compression-ratio-2-vqganr-double-l2-param-matched",
        "latent-slots": 32,
        "encoder-type": "vqganr",
    }

    assert make_run_name(parameters, "20260727") == (
        "compression-ratio-2-vqganr-double-l2-param-matched__20260727"
    )


def test_run_name_keeps_parameter_fallback_without_ablation():
    parameters = {
        "latent-slots": 32,
        "use-ema-codebook": True,
    }

    assert make_run_name(parameters, "20260727") == (
        "latent-slots-32__use-ema-codebook-true__20260727"
    )
