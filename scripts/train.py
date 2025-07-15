#!/usr/bin/env python3
"""
Training script for MultiPepGen.

Usage:
    python scripts/train.py --config configs/default_config.yaml
"""

import argparse
import yaml
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multipepgen.training.trainer import GANTrainer
from multipepgen.data.preprocessing import PeptidePreprocessor
from multipepgen.models.gan import ConditionalGAN
from multipepgen.models.generator import Generator
from multipepgen.models.discriminator import Discriminator


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train MultiPepGen model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--data-path', type=str, default='data/data_sample.csv',
                       help='Path to training data')
    parser.add_argument('--output-dir', type=str, default='results/models',
                       help='Output directory for trained models')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data preprocessor
    preprocessor = PeptidePreprocessor(
        sequence_length=config['data']['sequence_length'],
        vocab_size=config['data']['vocab_size']
    )
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_data, val_data = preprocessor.load_and_split_data(args.data_path)
    
    # Initialize models
    print("Initializing models...")
    generator = Generator(
        sequence_length=config['model']['sequence_length'],
        vocab_size=config['model']['vocab_size'],
        latent_dim=config['model']['latent_dim'],
        hidden_dims=config['model']['generator_hidden_dims'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    discriminator = Discriminator(
        sequence_length=config['model']['sequence_length'],
        vocab_size=config['model']['vocab_size'],
        hidden_dims=config['model']['discriminator_hidden_dims'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    # Initialize GAN
    gan = ConditionalGAN(
        generator=generator,
        discriminator=discriminator,
        latent_dim=config['model']['latent_dim'],
        sequence_length=config['model']['sequence_length'],
        vocab_size=config['model']['vocab_size']
    )
    
    # Initialize trainer
    trainer = GANTrainer(
        model=gan,
        config=config['training']
    )
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        output_dir=args.output_dir
    )
    
    print(f"Training completed! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main() 