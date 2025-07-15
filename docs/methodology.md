# MultiPepGen Methodology

## Overview

MultiPepGen employs a Conditional Generative Adversarial Network (CGAN) architecture to generate synthetic antimicrobial peptide sequences. The model learns the underlying patterns in known antimicrobial peptides and generates novel sequences with similar properties.

## Architecture

### Conditional GAN Structure

The model consists of two main components:

1. **Generator (G)**: Creates synthetic peptide sequences from random noise and conditional inputs
2. **Discriminator (D)**: Distinguishes between real and generated sequences

### Generator Architecture

```
Input: [Latent Vector (100d) + Condition (1d)] 
    ↓
Dense Layer (256 units) + ReLU + Dropout + BatchNorm
    ↓
Dense Layer (512 units) + ReLU + Dropout + BatchNorm
    ↓
Dense Layer (256 units) + ReLU + Dropout + BatchNorm
    ↓
Dense Layer (sequence_length × vocab_size) + Softmax
    ↓
Reshape: (sequence_length, vocab_size)
```

### Discriminator Architecture

```
Input: [Sequence (sequence_length, vocab_size) + Condition (1d)]
    ↓
Flatten: (sequence_length × vocab_size + 1)
    ↓
Dense Layer (256 units) + ReLU + Dropout + BatchNorm
    ↓
Dense Layer (128 units) + ReLU + Dropout + BatchNorm
    ↓
Dense Layer (64 units) + ReLU + Dropout
    ↓
Dense Layer (1 unit) + Sigmoid
```

## Training Process

### Loss Functions

- **Generator Loss**: Binary cross-entropy loss to fool the discriminator
- **Discriminator Loss**: Binary cross-entropy loss to distinguish real from fake

### Training Algorithm

1. **Discriminator Training**:
   - Sample real sequences from training data
   - Generate fake sequences using current generator
   - Train discriminator to classify real vs fake

2. **Generator Training**:
   - Generate fake sequences
   - Train generator to maximize discriminator's fake classification error

### Conditional Training

The model uses conditional inputs to control generation:
- Activity scores
- Sequence length preferences
- Source organism information

## Data Preprocessing

### Sequence Encoding

1. **Amino Acid Vocabulary**: 20 standard amino acids
2. **One-Hot Encoding**: Each position encoded as 20-dimensional vector
3. **Length Normalization**: All sequences padded/truncated to fixed length

### Data Augmentation

- Random sequence truncation
- Amino acid substitution with similar properties
- Sequence reversal for bidirectional training

## Evaluation Metrics

### Quality Metrics

1. **Sequence Validity**: Percentage of valid amino acid sequences
2. **Diversity**: Unique sequences generated
3. **Novelty**: Sequences not present in training data

### Biological Metrics

1. **Activity Prediction**: Using pre-trained activity predictors
2. **Physicochemical Properties**: Charge, hydrophobicity, etc.
3. **Structural Similarity**: Comparison with known antimicrobial peptides

## Hyperparameters

### Model Parameters

- **Latent Dimension**: 100
- **Sequence Length**: 50 amino acids
- **Vocabulary Size**: 20 amino acids
- **Dropout Rate**: 0.3

### Training Parameters

- **Learning Rate**: 0.0002 (Adam optimizer)
- **Batch Size**: 32
- **Epochs**: 100
- **Beta1**: 0.5, **Beta2**: 0.999

## Implementation Details

### Framework

- **Deep Learning**: TensorFlow 2.x / Keras
- **Data Processing**: NumPy, Pandas
- **Evaluation**: Scikit-learn

### Hardware Requirements

- **GPU**: Recommended for faster training
- **Memory**: 8GB+ RAM
- **Storage**: 10GB+ for models and data

## Limitations and Future Work

### Current Limitations

1. Fixed sequence length generation
2. Limited conditional control
3. No explicit biological constraints

### Future Improvements

1. Variable length sequence generation
2. Multi-objective optimization
3. Integration with molecular dynamics simulations
4. Real-time activity prediction

## References

1. Goodfellow, I., et al. (2014). Generative Adversarial Nets. NIPS.
2. Mirza, M., & Osindero, S. (2014). Conditional Generative Adversarial Nets. arXiv.
3. Wang, G., et al. (2016). APD3: the antimicrobial peptide database. NAR. 