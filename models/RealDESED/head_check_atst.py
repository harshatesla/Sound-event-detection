import torch
sd = torch.load(r"E:\workspace\audio_team\sed\models\RealDESED\resources\ATST-F_strong_1.pt", map_location="cpu", weights_only=True)
sd = sd.get("state_dict", sd)
for k, v in sd.items():
    if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == 447:
        print(k, tuple(v.shape))