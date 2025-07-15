# Data Directory

This directory contains the data files used for training and evaluation of the MultiPepGen model.

## File Structure

```
data/
├── raw/                    # Original, unprocessed data files
├── processed/              # Preprocessed data ready for training
└── README.md              # This file
```

## Data Format

The peptide sequence data should be in CSV format with the following columns:

- `sequence`: Amino acid sequence (string)
- `activity`: Antimicrobial activity score (float, optional)
- `length`: Sequence length (integer, optional)
- `source`: Data source identifier (string, optional)

## Sample Data

The `data_sample.csv` file contains 1000 example antimicrobial peptide sequences for demonstration purposes.

## Data Preprocessing

The data preprocessing pipeline:

1. **Sequence encoding**: Converts amino acid sequences to one-hot encoded format
2. **Length normalization**: Pads or truncates sequences to fixed length
3. **Data splitting**: Divides data into training and validation sets
4. **Conditional encoding**: Creates conditional inputs for the GAN

## Usage

```python
from multipepgen.data.preprocessing import PeptidePreprocessor

# Load and preprocess data
preprocessor = PeptidePreprocessor(sequence_length=50, vocab_size=20)
train_data, val_data = preprocessor.load_and_split_data("data/data_sample.csv")
```

## Data Sources

- **APD3**: Antimicrobial Peptide Database
- **DBAASP**: Database of Antimicrobial Activity and Structure of Peptides
- **CAMP**: Collection of Anti-Microbial Peptides

## Citation

If you use the provided sample data, please cite the original sources:

> Wang, G., Li, X., & Wang, Z. (2016). APD3: the antimicrobial peptide database as a tool for research and education. Nucleic acids research, 44(D1), D1087-D1093.

> Gogoladze, G., Grigolava, M., Vishnepolsky, B., Chubinidze, M., Duroux, P., Lefranc, M. P., & Pirtskhalava, M. (2014). DBAASP v. 2: an enhanced database of structure and antimicrobial/cytotoxic activity of natural and synthetic peptides. Nucleic acids research, 42(D1), D650-D654. 