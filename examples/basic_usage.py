"""
Basic usage example for MultiPepGen.

This example demonstrates how to:
1. Load example peptide data
2. Train a Conditional GAN model
3. Generate synthetic peptide sequences
4. Evaluate the generated sequences
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

from multipepgen.utils.preprocessing import preprocess_data
from multipepgen.validation.metrics import validation_scores
from multipepgen.models.cgan import ConditionalGAN
from multipepgen.config import LABELS


def main():
    """Main example function."""
    print("MultiPepGen Basic Usage Example")
    print("=" * 40)
    
    # Configuration
    labels = LABELS
    sequence_length = 35
    vocab_size = 21
    num_classes = 7
    latent_dim = 100
    batch_size = 32
    epochs = 1  # Reduced for example
    
    # Load sample data (you would use your actual data path)
    
    data_path = os.path.join(os.path.dirname(__file__), "data/data_sample.csv")
    train_data = pd.read_csv(data_path)
    print(f"\n Loading training samples: {len(train_data)}")
    dataset = preprocess_data(train_data, batch_size=batch_size)
    print(f"Preprocessing de data")
    
    # 2. Model initialization
    print("\n2. Initializing models...")    
    gan = ConditionalGAN(
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        latent_dim=latent_dim,
        num_classes = num_classes
    )
    
    # 3. Model compilation
    print("\n3. Compiling model...")
    gan.compile(
        d_optimizer=Adam(),
        g_optimizer=Adam(),
        loss_fn=BinaryCrossentropy()
    )
    
    # 4. Training (simplified for example)
    print("\n4. Training model...")
    gan.fit(dataset, epochs = epochs)
    
    # 5. Generate sequences
    print("\n5. Generating synthetic sequences...")
    num_sequences = 10
    
    generated_sequences = gan.generate_class_random(
        num_sequences=num_sequences
    )
    
    print(f"Generated {num_sequences} synthetic peptide sequences")
    print(f"Sequence shape: {generated_sequences.shape}")
    print("Show examples of sequences generated")
    print(generated_sequences.head(5))
    
    # 6. Evaluation
    print("\n6. Evaluating generated sequences...")
    scores, scores_df = validation_scores(train_data, generated_sequences)
    # Convert to amino acid sequences (simplified)
    # In practice, you would decode the one-hot encoded sequences
    print(f"Validations Scores: {scores}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main() 