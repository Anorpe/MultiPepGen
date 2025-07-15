# MultiPepGen

MultiPepGen is a neural network-based model designed to generate synthetic sequences of antimicrobial peptides with specific functionalities. It employs a Conditional GAN (Generative Adversarial Network) architecture with recurrent cells to create diverse and bioactive peptide sequences on demand.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/username/MultiPepGen.git
cd MultiPepGen

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Basic Usage

```python
from multipepgen import ConditionalGAN, PeptidePreprocessor

# Load and preprocess data
preprocessor = PeptidePreprocessor(sequence_length=50, vocab_size=20)
train_data, val_data = preprocessor.load_and_split_data("data/data_sample.csv")

# Initialize and train model
gan = ConditionalGAN(generator, discriminator)
# ... training code ...

# Generate synthetic sequences
sequences = gan.generate_sequences(num_sequences=100)
```

### Training

```bash
# Train with default configuration
python scripts/train.py --config configs/default_config.yaml

# Train with custom data
python scripts/train.py --config configs/default_config.yaml --data-path your_data.csv
```

## 📁 Project Structure

```
MultiPepGen/
├── src/multipepgen/          # Main package source code
│   ├── models/              # GAN architecture implementations
│   ├── data/                # Data preprocessing and loading
│   ├── training/            # Training utilities and callbacks
│   ├── evaluation/          # Evaluation metrics and visualization
│   └── utils/               # Utility functions
├── scripts/                 # Executable scripts
├── notebooks/               # Jupyter notebooks for exploration
├── configs/                 # Configuration files
├── tests/                   # Unit tests
├── examples/                # Usage examples
├── docs/                    # Documentation
└── results/                 # Output directory for models and results
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

## 📊 GUI Application

A web-based graphical user interface (GUI) for sequence generation and activity prediction is available at [MultiPepGen](https://multipepgen.medellin.unal.edu.co/)

## 📚 Documentation

- [API Reference](docs/api.md)
- [Methodology](docs/methodology.md)
- [Results and Benchmarks](docs/results.md)
- [Examples](examples/)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=multipepgen
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📖 Citing MultiPepGen

If you use this code or the GUI application in your research, please cite:

> Orrego, A. et al. (2025). *MultiPepGen: A Neural Network-Based Conditional GAN Model for Antimicrobial Peptide Sequence Generation*. [Journal Name, Volume(Issue), Pages]. DOI: [add when available]

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
