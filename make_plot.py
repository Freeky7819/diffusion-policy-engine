# make_plot.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("log.csv")

# 1) R_val čez epohe (bližje 0 je bolje)
plt.figure(figsize=(7,4))
plt.plot(df["epoch"], df["R_val"].astype(float))
plt.xlabel("Epoch")
plt.ylabel("R_val (higher is better, max ~ 0)")
plt.title("DPE: Reward vs Epoch")
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.savefig("results.png", dpi=160)
print("Saved results.png")

# 2) (opcijsko) loss čez epohe – odkomentiraj, če želiš še drugo sliko
# plt.figure(figsize=(7,4))
# plt.plot(df["epoch"], df["loss"].astype(float))
# plt.xlabel("Epoch")
# plt.ylabel("Training loss")
# plt.title("DPE: Loss vs Epoch")
# plt.grid(True, linewidth=0.3)
# plt.tight_layout()
# plt.savefig("loss.png", dpi=160)
# print("Saved loss.png")
