"""
MultiPepGen: A Neural Network-Based Conditional GAN Model for Antimicrobial Peptide Sequence Generation

This package provides tools for generating synthetic antimicrobial peptide sequences
using Conditional Generative Adversarial Networks (CGANs).
"""

__version__ = "1.0.0"
__author__ = "A. Orrego"
__email__ = "author@example.com"

from .models.cgan import ConditionalGAN

__all__ = [
    "ConditionalGAN"
] 