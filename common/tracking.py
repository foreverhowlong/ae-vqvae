"""Small, shared Weights & Biases integration for experiment entry points."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from dotenv import load_dotenv

from common import ROOT


DEFAULT_PROJECT = "ae-vqvae"


def init_wandb(name: str, *, config: dict[str, Any], group: str | None = None,
               tags: list[str] | None = None) -> Any:
    """Initialize a run after loading root-level environment settings."""
    load_dotenv(ROOT / ".env", override=False)
    import wandb

    return wandb.init(
        project=os.getenv("WANDB_PROJECT", DEFAULT_PROJECT),
        entity=os.getenv("WANDB_ENTITY"), name=name, group=group, tags=tags, config=config,
    )


@contextmanager
def wandb_run(
    name: str,
    *,
    config: dict[str, Any],
    group: str | None = None,
    tags: list[str] | None = None,
) -> Iterator[Any]:
    """Start and always finish one W&B run, loading credentials from root ``.env``."""
    run = init_wandb(name, config=config, group=group, tags=tags)
    try:
        yield run
    finally:
        run.finish()


def log(metrics: dict[str, Any], *, step: int | None = None) -> None:
    """Log metrics to the active run."""
    import wandb

    wandb.log(metrics, step=step)
