import time
import torch

from lerobot.policies.smolvla import SmolVLAPolicy

torch.set_num_threads(4)
device = torch.device("cpu")

print("Loading SmolVLA...")
t0 = time.time()

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy.to(device)
policy.eval()

print(f"Loaded successfully in {time.time() - t0:.2f}s")
print("device:", device)
print("policy type:", type(policy))