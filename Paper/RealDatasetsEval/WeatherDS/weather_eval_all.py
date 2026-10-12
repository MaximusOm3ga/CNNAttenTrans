import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_quantformer_weather import evaluate_quantformer_weather
from eval_patchtst_weather import evaluate_patchtst_weather
from eval_tcn_weather import evaluate_tcn_weather
from train_patchtst_weather import train_patchtst_weather
from train_quantformer_weather import train_quantformer_weather
from train_tcn_weather import train_tcn_weather

HORIZONS = [96, 192, 336, 720]


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models on WeatherDS for multiple horizons.")
    parser.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    horizons = args.horizons
    for horizon in horizons:
        print(f"\n=== Forecast horizon {horizon} ===")
        if not args.skip_train:
            print("Training QuantFormer...")
            train_quantformer_weather(horizon=horizon, epochs=args.epochs, batch_size=args.batch_size)
            print("Training TCN...")
            train_tcn_weather(horizon=horizon, epochs=args.epochs, batch_size=args.batch_size)
            print("Training PatchTST...")
            train_patchtst_weather(horizon=horizon, epochs=args.epochs, batch_size=args.batch_size)

        print("Evaluating QuantFormer...")
        evaluate_quantformer_weather(horizon=horizon, batch_size=args.batch_size)
        print("Evaluating TCN...")
        evaluate_tcn_weather(horizon=horizon, batch_size=args.batch_size)
        print("Evaluating PatchTST...")
        evaluate_patchtst_weather(horizon=horizon, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
