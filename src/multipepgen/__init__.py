"""
MultiPepGen: A Neural Network-Based Conditional GAN Model for Antimicrobial Peptide Sequence Generation

This package provides tools for generating synthetic antimicrobial peptide sequences
using Conditional Generative Adversarial Networks (CGANs).
"""

__version__ = "1.0.0"
__author__ = "A. Orrego"
__email__ = "author@example.com"

from .models.gan import ConditionalGAN
from .data.preprocessing import PeptidePreprocessor
from .training.trainer import GANTrainer
from .evaluation.metrics import PeptideEvaluator

__all__ = [
    "ConditionalGAN",
    "PeptidePreprocessor", 
    "GANTrainer",
    "PeptideEvaluator",
] 