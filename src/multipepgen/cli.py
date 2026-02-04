import argparse
import os
import sys
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from multipepgen.models.cgan import ConditionalGAN
from multipepgen.utils.preprocessing import preprocess_data
from multipepgen.validation.metrics import validation_scores
from multipepgen.config import LABELS
from multipepgen.utils.logger import logger

def load_config(config_path):
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} not found.")
        sys.exit(1)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train(args):
    config = load_config(args.config)
    
    # Extract params from config
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})
    
    if not os.path.exists(args.data):
        logger.error(f"Data file {args.data} not found.")
        sys.exit(1)
        
    logger.info(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    dataset = preprocess_data(
        df, 
        batch_size=train_cfg.get('batch_size', 32),
        max_len=data_cfg.get('sequence_length', 35)
    )
    
    logger.info("Initializing model...")
    gan = ConditionalGAN(
        sequence_length=model_cfg.get('sequence_length', 35),
        vocab_size=model_cfg.get('vocab_size', 21),
        latent_dim=model_cfg.get('latent_dim', 100),
        num_classes=len(LABELS)
    )
    
    logger.info("Compiling model...")
    gan.compile(
        d_optimizer=Adam(learning_rate=train_cfg.get('learning_rate_discriminator', 0.0002)),
        g_optimizer=Adam(learning_rate=train_cfg.get('learning_rate_generator', 0.0002)),
        loss_fn=BinaryCrossentropy()
    )
    
    logger.info(f"Starting training for {train_cfg.get('epochs', 100)} epochs...")
    gan.fit(dataset, epochs=train_cfg.get('epochs', 100))
    
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        gan.save_model_weights(args.output)
        logger.info(f"Model weights saved to {args.output}_*.weights.h5")

def generate(args):
    config = load_config(args.config)
    model_cfg = config.get('model', {})
    
    logger.info("Initializing model...")
    gan = ConditionalGAN(
        sequence_length=model_cfg.get('sequence_length', 35),
        vocab_size=model_cfg.get('vocab_size', 21),
        latent_dim=model_cfg.get('latent_dim', 100),
        num_classes=len(LABELS)
    )
    
    if args.weights:
        logger.info(f"Loading weights from {args.weights}...")
        gan.load_model_weights(args.weights)
    else:
        logger.warning("No weights provided. Generating with untrained model.")
        
    logger.info(f"Generating {args.num} sequences...")
    if args.classes:
        # Expect classes as comma-separated list
        classes_list = args.classes.split(',')
        logger.info(f"Target classes: {classes_list}")
        generated_df = gan.generate_class(args.num, classes_list)
    else:
        logger.info("Generating with random class assignments.")
        generated_df = gan.generate_class_random(args.num)
        
    if args.output:
        generated_df.to_csv(args.output, index=False)
        logger.info(f"Generated sequences saved to {args.output}")
    else:
        logger.info("\nGenerated Sequences:")
        logger.info(generated_df.to_string())

def main():
    parser = argparse.ArgumentParser(description="MultiPepGen CLI - Generate peptides using CGAN")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--data", required=True, help="Path to training CSV data")
    train_parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config YAML")
    train_parser.add_argument("--output", help="Prefix for saving model weights")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate new sequences")
    gen_parser.add_argument("--num", type=int, default=10, help="Number of sequences to generate")
    gen_parser.add_argument("--weights", help="Prefix of the weights to load")
    gen_parser.add_argument("--classes", help="Comma-separated list of target classes (e.g. 'microbiano,cancer')")
    gen_parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config YAML")
    gen_parser.add_argument("--output", help="Output CSV file for generated sequences")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "generate":
        generate(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
