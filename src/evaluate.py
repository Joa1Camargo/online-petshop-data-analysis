import numpy as np


def compute_error(predicted, actual) -> dict:
    error = predicted - actual
    total_y, ape = 0, 0
    for y, p in zip(actual, predicted):
        if y > 0:
            ape += abs((y - p) / y)
            total_y += 1
    mape = ape / total_y
    results = {"MAPE": mape, "MAE": np.mean(np.abs(error))}
    print("=" * 30)
    print(f"{'ERROR METRICS':^30}")
    print("=" * 30)
    print(f"  MAPE: {mape:.2%}")
    print(f"  MAE:  {results['MAE']:,.2f}")
    print("=" * 30)
    return results