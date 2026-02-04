import argparse
import pandas as pd
import json
from multipepgen.validation.metrics import validation_scores
from multipepgen.utils.logger import logger

def evaluate_sequences(ref_path, gen_path, output_json=None):
    """
    Evaluates generated sequences against a reference dataset and saves scores.
    """
    logger.info(f"Loading reference data from {ref_path}...")
    ref_df = pd.read_csv(ref_path)
    
    logger.info(f"Loading generated data from {gen_path}...")
    gen_df = pd.read_csv(gen_path)
    
    logger.info("Calculating validation scores...")
    scores, details = validation_scores(ref_df, gen_df)
    
    if output_json:
        # Convert non-serializable objects (like lists of columns) to strings if needed
        serializable_scores = {}
        for k, v in scores.items():
            if isinstance(v, (float, int, str)):
                serializable_scores[k] = v
            else:
                serializable_scores[k] = str(v)
                
        with open(output_json, 'w') as f:
            json.dump(serializable_scores, f, indent=4)
        logger.info(f"Scores saved to {output_json}")
        
    logger.info("\nValidation Scores Summary:")
    for k, v in scores.items():
        logger.info(f"  {k}: {v}")
        
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiPepGen individual sequence evaluation")
    parser.add_argument("--ref", required=True, help="Path to reference CSV")
    parser.add_argument("--gen", required=True, help="Path to generated CSV")
    parser.add_argument("--output", help="Optional path to save scores as JSON")
    
    args = parser.parse_args()
    evaluate_sequences(args.ref, args.gen, args.output)
