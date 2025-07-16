"""
Basic usage example for MultiPepGen.

This example demonstrates how to:
1. Load example peptide data
2. Train a Conditional GAN model
3. Generate synthetic peptide sequences
4. Evaluate the generated sequences
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multipepgen.models.gan import ConditionalGAN
from multipepgen.models.generator import Generator
from multipepgen.models.discriminator import Discriminator


def main():
    """Main example function."""
    print("MultiPepGen Basic Usage Example")
    print("=" * 40)
    
    # Configuration
    sequence_length = 50
    vocab_size = 20
    latent_dim = 100
    batch_size = 32
    epochs = 10  # Reduced for example
    
    # Load sample data (you would use your actual data path)
    data_path = "../data/data_sample.csv"
    try:
        train_data = pd.read_csv(data_path)
        print(f"Training samples: {len(train_data)}")
    except FileNotFoundError:
        print("Sample data not found. Using dummy data for demonstration.")
        # Create dummy data for demonstration
        train_data = np.random.rand(100, sequence_length, vocab_size)
    
    # 2. Model initialization
    print("\n2. Initializing models...")
    generator = Generator(
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        latent_dim=latent_dim,
        hidden_dims=(256, 512, 256),
        dropout_rate=0.3
    )
    
    discriminator = Discriminator(
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        hidden_dims=(256, 128, 64),
        dropout_rate=0.3
    )
    
    gan = ConditionalGAN(
        generator=generator,
        discriminator=discriminator,
        latent_dim=latent_dim,
        sequence_length=sequence_length,
        vocab_size=vocab_size
    )
    
    # 3. Model compilation
    print("\n3. Compiling model...")
    gan.compile(
        g_optimizer="adam",
        d_optimizer="adam",
        g_loss_fn="binary_crossentropy",
        d_loss_fn="binary_crossentropy"
    )
    
    # 4. Training (simplified for example)
    print("\n4. Training model...")
    print("Note: This is a simplified training loop for demonstration.")
    print("In practice, you would use the GANTrainer class.")
    
    # Simple training loop
    for epoch in range(epochs):
        # Sample batch
        batch_size_actual = min(batch_size, len(train_data))
        batch_indices = np.random.choice(len(train_data), batch_size_actual, replace=False)
        batch_data = train_data[batch_indices]
        
        # Create dummy conditions
        conditions = np.random.randn(batch_size_actual, 1)
        
        # Train step
        loss = gan.train_on_batch([batch_data, conditions])
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
    
    # 5. Generate sequences
    print("\n5. Generating synthetic sequences...")
    num_sequences = 10
    conditions = np.random.randn(num_sequences, 1)
    
    generated_sequences = gan.generate_sequences(
        num_sequences=num_sequences,
        conditions=conditions,
        temperature=1.0
    )
    
    print(f"Generated {num_sequences} synthetic peptide sequences")
    print(f"Sequence shape: {generated_sequences.shape}")
    
    # 6. Evaluation
    print("\n6. Evaluating generated sequences...")
    
    # Convert to amino acid sequences (simplified)
    # In practice, you would decode the one-hot encoded sequences
    print("Generated sequences (first 5):")
    for i in range(min(5, num_sequences)):
        # Simplified: just show the shape
        print(f"Sequence {i+1}: shape {generated_sequences[i].shape}")
    
    print("\nExample completed successfully!")
    print("\nNext steps:")
    print("- Use the full GANTrainer for proper training")
    print("- Implement proper sequence decoding")
    print("- Add more sophisticated evaluation metrics")
    print("- Save and load trained models")


if __name__ == "__main__":
    main() 