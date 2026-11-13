"""Review: D scaling c* at which the genemean infinite-seed pb_rmse projection reaches the best baseline
(c=1: observed seed pair; MSE_inf(c) = MSE_single - c*D/2). Writes nothing; prints."""
from review_arms_common import *  # noqa
A = json.loads((OUT / "review_arms_analysis.json").read_text())
for panel in ARMS:
    for f in range(3):
        z = np.load(OUT / f"review_arms_profiles_{panel}_fold{f}.npz")
        held = z["held"]; obs = z["obs"][held].astype(float)
        p0 = z["genemean|0|P"][held].astype(float); p1 = z["genemean|1|P"][held].astype(float)
        ms = (((p0 - obs) ** 2).mean(1) + ((p1 - obs) ** 2).mean(1)) / 2; D = ((p0 - p1) ** 2).mean(1)
        bl = A[f"B|genemean|{panel}|{f}"]["best_baseline"][1]
        cs = np.linspace(0.5, 2, 301)
        v = [np.sqrt(np.maximum(ms - c * D / 2, 0)).mean() for c in cs]
        cstar = next((c for c, x in zip(cs, v) if x <= bl), None)
        print(f"genemean {panel} fold{f}: c* = {cstar:.2f}  inf(c=1.25) = {np.sqrt(np.maximum(ms - 1.25 * D / 2, 0)).mean():.3f} baseline {bl:.3f}")
