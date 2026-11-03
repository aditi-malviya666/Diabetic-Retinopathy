"""M9/M10: serve the API and the dashboard.

Uses whatever models exist under --project: checkpoints/m6_<preset>/best.pt with
reports/m6_<preset>/m6_results.json (triage), checkpoints/m5_<preset>/final.pt
(vessels, artery/vein and physics). Missing parts are reported, not faked.

    python scripts/m9_serve.py --project .            then open http://127.0.0.1:8000
    curl -F image=@photo.jpg http://127.0.0.1:8000/api/analyse
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_host = os.environ.get("HOST", "0.0.0.0" if ("PORT" in os.environ or "SPACE_ID" in os.environ) else "127.0.0.1")
    default_port = int(os.environ.get("PORT", 7860 if "SPACE_ID" in os.environ else 8000))
    
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=str(root))
    ap.add_argument("--host", default=default_host)
    ap.add_argument("--port", type=int, default=default_port)
    ap.add_argument("--triage-preset", default="full", choices=["full", "quick"])
    ap.add_argument("--seg-preset", default="full", choices=["full", "quick"])
    ap.add_argument("--mc-samples", type=int, default=50)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import uvicorn

    from drpinn.api.app import create_app
    from drpinn.api.service import ServiceConfig

    cfg = ServiceConfig(project=Path(a.project), triage_preset=a.triage_preset, seg_preset=a.seg_preset,
                        device=a.device, mc_samples=a.mc_samples)
    print(f"Dashboard: http://{a.host}:{a.port}/   (research prototype, not a medical device)")
    uvicorn.run(create_app(config=cfg), host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
