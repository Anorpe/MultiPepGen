import pandas as pd
import json
import os
from multipepgen.validation.metrics import validation_scores
from multipepgen.utils.logger import logger

def compare_models(reference_data, model_results_dict, output_file="benchmark_results.json"):
    """
    Compares multiple models against a reference dataset.
    
    Args:
        reference_data (pd.DataFrame): The original training/experimental data.
        model_results_dict (dict): Keys are model names, values are DataFrames of generated sequences.
        output_file (str): Path to save the comparison results.
    """
    all_results = {}
    
    for model_name, generated_df in model_results_dict.items():
        logger.info(f"Evaluating {model_name}...")
        scores, _ = validation_scores(reference_data, generated_df)
        all_results[model_name] = scores
        
    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    logger.info(f"Benchmarking completed. Results saved to {output_file}")
    
    # Print summary table
    summary_df = pd.DataFrame(all_results).T
    logger.info("\nBenchmark Summary Table:")
    logger.info("\n" + summary_df.to_string())
    
    return summary_df

if __name__ == "__main__":
    # Example usage for CLI benchmarking
    import argparse
    parser = argparse.ArgumentParser(description="MultiPepGen Benchmarking Tool")
    parser.add_argument("--ref", required=True, help="Path to reference CSV")
    parser.add_argument("--models", required=True, help="Comma-separated paths to model output CSVs")
    parser.add_argument("--names", help="Comma-separated names for the models")
    parser.add_argument("--output", default="benchmark_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    ref_df = pd.read_csv(args.ref)
    model_paths = args.models.split(',')
    model_names = args.names.split(',') if args.names else [os.path.basename(p) for p in model_paths]
    
    model_dict = {}
    for name, path in zip(model_names, model_paths):
        model_dict[name] = pd.read_csv(path)
        
    compare_models(ref_df, model_dict, args.output)
