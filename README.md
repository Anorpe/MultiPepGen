# MultiPepGen

MultiPepGen is a neural network-based model designed to generate synthetic sequences of antimicrobial peptides with specific functionalities. It employs a Conditional GAN (Generative Adversarial Network) architecture with recurrent cells to create diverse and bioactive peptide sequences on demand.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/anorpe/MultiPepGen.git
cd MultiPepGen

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Basic Usage

```python
import pandas as pd
from multipepgen.models.cgan import ConditionalGAN
from multipepgen.utils.preprocessing import preprocess_data
from multipepgen.validation.metrics import validation_scores
from multipepgen.config import LABELS

# Configuration
sequence_length = 35
vocab_size = 21
num_classes = 7
latent_dim = 100
batch_size = 32

# Load and preprocess example data
data_path = "examples/data/data_sample.csv"
df = pd.read_csv(data_path)
dataset = preprocess_data(df, batch_size=batch_size)

# Model initialization and compilation
gan = ConditionalGAN(
    sequence_length=sequence_length,
    vocab_size=vocab_size,
    latent_dim=latent_dim,
    num_classes=num_classes
)
gan.compile(
    d_optimizer="adam",
    g_optimizer="adam",
    loss_fn="binary_crossentropy"
)

# Training (adjust epochs as needed)
gan.fit(dataset, epochs=10)

# Generate synthetic sequences
num_sequences = 10
generated_sequences = gan.generate_class_random(num_sequences=num_sequences)
print(generated_sequences.head())

# Evaluation of generated sequences
scores, scores_df = validation_scores(df, generated_sequences)
print(scores)
```

## 📁 Project Structure

```
MultiPepGen/
├── src/multipepgen/          # Main source code
│   ├── models/               # Conditional GAN architecture implementation
│   │   └── cgan.py           # Main ConditionalGAN model
│   ├── utils/                # Utility functions (preprocessing, postprocessing, descriptors)
│   ├── validation/           # Sequence metrics and validation
│   ├── config.py             # Configuration and labels
│   └── __init__.py
├── examples/                 # Usage example and sample data
│   ├── basic_usage.py        # Example script
│   └── data/
│       └── data_sample.csv   # Sample data
├── configs/                  # Configuration files
│   └── default_config.yaml
├── docs/                     # Documentation and figures
│   ├── methodology.md
│   └── figures/
├── tests/                    # (Empty) Space for unit tests
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── setup.py                  # Installation
├── pyproject.toml            # Build configuration
└── LICENSE                   # License
```

## 🔧 Requirements

Main dependencies:
- numpy
- pandas
- scikit-learn
- tensorflow
- keras
- pyyaml
- matplotlib
- seaborn
- plotly
- tqdm
- biopython
- joblib
- imageio
- xgboost
- modlamp

For the complete list and recommended versions, see `requirements.txt`.

## Recommended Environment

It is recommended to use **Python 3.10** to ensure compatibility with TensorFlow and all project dependencies. More recent versions of Python may not be compatible with some scientific libraries.

## 📊 Web Application (GUI)

A web-based graphical user interface for sequence generation and activity prediction is available at: [MultiPepGen](https://multipepgen.medellin.unal.edu.co/)

## 📚 Documentation

- [Methodology](docs/methodology.md)
- [Usage Example](examples/basic_usage.py)

## 🧪 Testing

Currently, there are no tests implemented in the `tests/` folder, but you can add your own tests following the standard pytest structure.


## 📖 Citing MultiPepGen

If you use this code or the web application in your research, please cite:

> Orrego, A. et al. (2025). *MultiPepGen: A Neural Network-Based Conditional GAN Model for Antimicrobial Peptide Sequence Generation*. [Journal Name, Volume(Issue), Pages]. DOI: [to be added when available]

**BibTeX:**
```bibtex
@article{orrego2025multipepgen,
  title={MultiPepGen: A Neural Network-Based Conditional GAN Model for Antimicrobial Peptide Sequence Generation},
  author={Orrego, A. and others},
  journal={Journal Name},
  year={2025},
  volume={XX},
  number={X},
  pages={XX--XX},
  doi={DOI}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
