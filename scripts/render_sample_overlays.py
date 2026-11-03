import base64
from pathlib import Path
from PIL import Image
import numpy as np

import sys
sys.path.insert(0, "dr-pinn/src")

from drpinn.api.service import Analyser, ServiceConfig

cfg = ServiceConfig(project=Path("dr-pinn"), mc_samples=10)
analyser = Analyser(cfg)

samples_dir = Path("dr-pinn/src/drpinn/api/static/samples")

for filename, prefix in [("sample1_healthy.png", "sample1"), ("sample2_severe_dr.png", "sample2"), ("sample3_stenosis.png", "sample3")]:
    img_path = samples_dir / filename
    if not img_path.exists(): continue
    img = np.array(Image.open(img_path).convert("RGB"))
    res = analyser.analyse(img, mc_samples=5)
    if res.get("vessels", {}).get("overlay_png"):
        (samples_dir / f"{prefix}_vessels.png").write_bytes(base64.b64decode(res["vessels"]["overlay_png"]))
    if res.get("shear", {}).get("overlay_png"):
        (samples_dir / f"{prefix}_wss.png").write_bytes(base64.b64decode(res["shear"]["overlay_png"]))
    if res.get("triage", {}).get("gradcam_png"):
        (samples_dir / f"{prefix}_gradcam.png").write_bytes(base64.b64decode(res["triage"]["gradcam_png"]))
    print(f"{prefix} overlays updated with transparent masks.")
