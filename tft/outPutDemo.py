from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tft.predictor import TFTPredictor  # noqa: E402


def main() -> None:
    bundle = Path("tft/artifacts/tft_bundle.pt")
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle not found at {bundle}. Train first or update the path.")

    pred = TFTPredictor(bundle_path=str(bundle))

    request_payload = {
        "tmc": "101N04923",  # segment ID
        "horizons": [5, 15, 30, 60],
        "ts": datetime.now(timezone.utc),
    }

    out = pred.predict(
        tmc=request_payload["tmc"],
        ts=request_payload["ts"],
        horizons=request_payload["horizons"],
        live=False,  # offline dataset cached next to the bundle
    )

    response = {
        "backend": "tft",
        "tmc": request_payload["tmc"],
        "ts_utc": request_payload["ts"].isoformat(),
        "speed_ratio": {str(h): float(v) for h, v in out.items()},
    }

    print("Request:")
    print(request_payload)
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
