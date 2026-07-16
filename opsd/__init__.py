"""
On-Policy Self-Distillation (OPSD) using TRL.

This module implements Self-Distillation Fine-Tuning (SDFT) based on
"Self-Distillation Enables Continual Learning" (https://arxiv.org/abs/2601.19897)

Reference implementation: https://github.com/idanshen/Self-Distillation

Key concept:
- Teacher: Same model that sees the prompt + gold answer demonstration in context
- Student: Same model that sees only the prompt
- Completions are GENERATED ON-POLICY during training
- Student learns to match teacher's distribution via KL divergence

This enables learning from demonstrations without catastrophic forgetting.

DistilTrainer: Full-featured, extends transformers.Trainer (recommended)
"""

from .trainer import DistilTrainer, prepare_distil_dataset
from .config import DistilConfig

__all__ = [
    "DistilTrainer",
    "DistilConfig", 
    "prepare_distil_dataset",
]
