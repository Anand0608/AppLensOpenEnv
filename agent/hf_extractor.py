"""HuggingFace-backed feature extractor for Stable-Baselines3.

Converts the 8-dimensional observation vector back into a natural-language
description of the current agent state, tokenises it with a pre-trained
DistilBERT model from HuggingFace, and returns the [CLS] embedding as the
rich feature representation that the SB3 PPO policy network then acts on.

This grounds the policy in a language model's world-knowledge about what
each action name means — giving significantly better sample efficiency than
a plain MLP on the raw binary vector.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from transformers import AutoModel, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.mock_env import REQUIRED_ACTIONS

HF_MODEL_NAME = "distilbert-base-uncased"


def _obs_to_text(obs_vec: list[float]) -> str:
    """Convert the float32 observation vector to a readable sentence."""
    flags = obs_vec[: len(REQUIRED_ACTIONS)]
    step_frac = obs_vec[-1]

    completed = [a for a, f in zip(REQUIRED_ACTIONS, flags) if f > 0.5]
    remaining = [a for a, f in zip(REQUIRED_ACTIONS, flags) if f <= 0.5]

    completed_str = ", ".join(completed) if completed else "none"
    remaining_str = ", ".join(remaining) if remaining else "none"
    step_pct = int(round(step_frac * 100))

    return (
        f"Completed actions: {completed_str}. "
        f"Remaining actions: {remaining_str}. "
        f"Progress: {step_pct}%."
    )


class HFDistilBertExtractor(BaseFeaturesExtractor):
    """Stable-Baselines3 feature extractor backed by DistilBERT.

    Parameters
    ----------
    observation_space:
        Gymnasium Box(8,) passed in by SB3 automatically.
    features_dim:
        Output size of the linear projection head (default 128).
    hf_model_name:
        HuggingFace model identifier.
    freeze_bert:
        When True (default) the BERT weights are frozen — only the
        projection head is trained.  Set False to fine-tune end-to-end.
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 128,
        hf_model_name: str = HF_MODEL_NAME,
        freeze_bert: bool = True,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        self.bert = AutoModel.from_pretrained(hf_model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size  # 768 for DistilBERT
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
        )

    # ------------------------------------------------------------------
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """observations: (batch, 8) float32 tensor from SB3."""
        device = observations.device
        texts = [_obs_to_text(row.tolist()) for row in observations]

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.bert.parameters())):
            out = self.bert(**encoded)

        cls_embed = out.last_hidden_state[:, 0, :]  # (batch, 768)
        return self.projection(cls_embed)             # (batch, features_dim)
