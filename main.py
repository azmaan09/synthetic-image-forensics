"""
Main entrypoint for the Synthetic AI Image Detection project.
Runs training and evaluation pipelines from the repository root.
"""

import argparse
from pathlib import Path

from src.train_cnn import train_cnn
from src.train_feature_model import train_feature_models
from src.evaluate import evaluate_all
from src.faithfulness_analysis import run_faithfulness_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic AI Image Detection pipeline")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Outputs directory")
    parser.add_argument("--skip-train-cnn", action="store_true", help="Skip CNN training")
    parser.add_argument("--skip-train-features", action="store_true", help="Skip feature model training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-faithfulness", action="store_true", help="Skip faithfulness analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_train_cnn:
        train_cnn(data_dir=str(data_dir), outputs_dir=str(outputs_dir))

    if not args.skip_train_features:
        train_feature_models(data_dir=str(data_dir), outputs_dir=str(outputs_dir))

    if not args.skip_eval:
        evaluate_all(data_dir=str(data_dir), outputs_dir=str(outputs_dir))

    if not args.skip_faithfulness:
        run_faithfulness_analysis(data_dir=str(data_dir), outputs_dir=str(outputs_dir))


if __name__ == "__main__":
    main()
