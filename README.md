# MultiPepGen

MultiPepGen is a neural network-based model designed to generate synthetic sequences of antimicrobial peptides with specific functionalities. It employs a Conditional GAN (Generative Adversarial Network) architecture with recurrent cells to create diverse and bioactive peptide sequences on demand.

## Requirements

To install the necessary libraries, use:

```bash
pip install -r requirements.txt
```

Main dependencies:
- numpy
- pandas
- scikit-learn
- tensorflow
- keras

## Contents

- `data/data_sample.csv` — Example dataset of antimicrobial peptide sequences.
- `train_model.ipynb` — Jupyter notebook for model training and evaluation.
- `utils.py` — Utility functions for preprocessing and model management.
- `requirements.txt` — List of required Python libraries.

## GUI Application

A web-based graphical user interface (GUI) for sequence generation and activity prediction is available at [MultiPepGen](https://multipepgen.medellin.unal.edu.co/)

## Citing MultiPepGen

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

## License

[Specify license here if applicable — MIT, GPLv3, Apache 2.0, etc.]
