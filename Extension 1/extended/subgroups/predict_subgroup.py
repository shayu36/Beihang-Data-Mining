"""Predict one six-column row with the saved conditional-calibration model."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--values", type=float, nargs=6, required=True, metavar="VALUE")
    args = parser.parse_args()
    x = np.asarray([args.values], dtype=float)
    model = joblib.load(args.model)
    result = model.predict(x)
    print(json.dumps(dict(raw_score=float(model.raw_score(x)[0]),
                          multiplier=float(model.multipliers(x)[0]),
                          score=float(result["score"][0]),
                          threshold=result["threshold"],
                          prediction=int(result["prediction"][0])), indent=2))
