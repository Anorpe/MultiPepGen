import os
import yaml

# Hardcoded constants for the biological domain
LABELS = ['microbiano', 'bacteriano', 'antigramneg', 'antigrampos', 'fungico', 'viral', 'cancer'] 

# List of valid amino acids (includes '_')
VALID_AMINOACIDS = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V', '_'
]
SET_VALID_AMINOACIDS = set(VALID_AMINOACIDS) 

# Configuration loading logic
def _get_default_config():
    """Returns basic defaults if no config file is found."""
    return {
        "data": {"sequence_length": 35},
        "model": {"latent_dim": 100},
        "training": {"batch_size": 32, "epochs": 100}
    }

def load_config(path=None):
    if path is None:
        # Try to find default config in the package
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, "..", "configs", "default_config.yaml")
        
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return _get_default_config()

# Global configuration object
DEFAULT_CONFIG = load_config()