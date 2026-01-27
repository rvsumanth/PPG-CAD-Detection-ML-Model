import argparse
from src.inference import predict_only, evaluate_unseen

parser = argparse.ArgumentParser(
    description="CAD Prediction & Evaluation using trained PPG model"
)

parser.add_argument(
    "--mode",
    choices=["predict", "evaluate"],
    required=True,
    help="predict = no labels | evaluate = unseen data with labels"
)

parser.add_argument(
    "--input",
    required=True,
    help="Path to input CSV file"
)

args = parser.parse_args()

MODEL_PATH = "models/CAD_Harmonized_91_77_Model.pkl"

if args.mode == "predict":
    predict_only(
        csv_path=args.input,
        model_path=MODEL_PATH
    )

elif args.mode == "evaluate":
    evaluate_unseen(
        csv_path=args.input,
        model_path=MODEL_PATH
    )

