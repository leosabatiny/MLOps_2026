from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

csv_path = Path("outputs/q9/losses.csv")
out_path = Path("assets/q9_loss_curve.png")
out_path.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

# Use epoch-level summaries (step == 0 rows)
train = df[df["split"] == "train_epoch"].sort_values("epoch")
val = df[df["split"] == "val_epoch"].sort_values("epoch")

plt.figure(figsize=(7, 4))
plt.plot(train["epoch"], train["loss"], label="train loss")
plt.plot(val["epoch"], val["loss"], label="val loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("PCAM: loss curves (epoch averages)")
plt.legend()
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
