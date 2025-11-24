import torch
import numpy as np

# this file was to test after an error
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

bundle = torch.load("../code/best_nn_model.pt", map_location="cpu", weights_only=False)

print("Loaded OK. Keys in checkpoint:")
print(bundle.keys())

