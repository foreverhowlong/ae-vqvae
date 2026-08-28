"""Single-source phase schedule for end-to-end tokenizer curricula."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EndToEndPhase:
    name: str
    use_quantizer: bool
    train_tokenizer: bool
    train_text_decoder: bool
    train_prior: bool
    update_codebook: bool
    prior_weight: float
    segmenter_downstream_weight: float


@dataclass(frozen=True)
class EndToEndPhaseSchedule:
    ae_warmup_steps: int = 0
    vq_warmup_steps: int = 0
    prior_catchup_steps: int = 0
    prior_anneal_steps: int = 0
    segmenter_only_downstream: bool = False
    segmenter_downstream_weight: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.ae_warmup_steps,
            self.vq_warmup_steps,
            self.prior_catchup_steps,
            self.prior_anneal_steps,
            self.segmenter_downstream_weight,
        ) < 0:
            raise ValueError("Phase durations and weights must be non-negative.")
        if self.prior_catchup_steps and not self.segmenter_only_downstream:
            raise ValueError(
                "prior_catchup_steps requires segmenter_only_downstream."
            )

    @property
    def reserved_steps(self) -> int:
        return (
            self.ae_warmup_steps
            + self.vq_warmup_steps
            + self.prior_catchup_steps
            + self.prior_anneal_steps
        )

    def state(self, optimizer_step: int) -> EndToEndPhase:
        if optimizer_step < 1:
            raise ValueError("optimizer_step is one-indexed and must be positive.")
        cursor = self.ae_warmup_steps
        if optimizer_step <= cursor:
            return EndToEndPhase(
                "ae_warmup",
                use_quantizer=False,
                train_tokenizer=True,
                train_text_decoder=True,
                train_prior=False,
                update_codebook=False,
                prior_weight=0.0,
                segmenter_downstream_weight=0.0,
            )

        cursor += self.vq_warmup_steps
        if optimizer_step <= cursor:
            return EndToEndPhase(
                "vq_warmup",
                use_quantizer=True,
                train_tokenizer=True,
                train_text_decoder=True,
                train_prior=False,
                update_codebook=True,
                prior_weight=0.0,
                segmenter_downstream_weight=0.0,
            )

        cursor += self.prior_catchup_steps
        if optimizer_step <= cursor:
            return EndToEndPhase(
                "prior_catchup",
                use_quantizer=True,
                train_tokenizer=False,
                train_text_decoder=False,
                train_prior=True,
                update_codebook=False,
                prior_weight=1.0,
                segmenter_downstream_weight=0.0,
            )

        anneal_start = cursor
        cursor += self.prior_anneal_steps
        if optimizer_step <= cursor:
            prior_weight = (
                optimizer_step - anneal_start
            ) / max(self.prior_anneal_steps, 1)
            return EndToEndPhase(
                "prior_anneal",
                use_quantizer=True,
                train_tokenizer=True,
                train_text_decoder=True,
                train_prior=True,
                update_codebook=True,
                prior_weight=prior_weight,
                segmenter_downstream_weight=(
                    self.segmenter_downstream_weight * prior_weight
                    if self.segmenter_only_downstream
                    else 0.0
                ),
            )

        return EndToEndPhase(
            "joint",
            use_quantizer=True,
            train_tokenizer=True,
            train_text_decoder=True,
            train_prior=True,
            update_codebook=True,
            prior_weight=1.0,
            segmenter_downstream_weight=(
                self.segmenter_downstream_weight
                if self.segmenter_only_downstream
                else 0.0
            ),
        )
