import torch
sd = torch.load(r"E:\workspace\audio_team\sed\models\RealDESED\runs\experiment_dumps\checkpoints\best-9.ckpt", map_location="cpu")

if isinstance(sd, dict):
    print("Top-level keys:", list(sd.keys()))
    state_dict = sd.get("state_dict", sd)
else:
    state_dict = sd

# 2. Print layer names and shapes (no weight values)
print("\n--- Layer Architecture (Name -> Shape) ---")
for key, value in state_dict.items():
    if hasattr(value, "shape"):
        print(f"{key:<60} {tuple(value.shape)}")

sd = sd.get("state_dict", sd)
for k, v in sd.items():
    if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] in (15, 447):
        print(k, tuple(v.shape))

