import torch
from model import ViT
from train import SyntheticImages

model = ViT()  # use the same --dim/--depth/--heads you trained with
model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
model.eval()

images = torch.stack([img for img, label in SyntheticImages(15, offset=100_000)])

# images: (B, 3, 32, 32), normalised with the CIFAR-10 mean/std from train.py
with torch.no_grad():
    logits = model(images)        # (B, 10)
probs = logits.softmax(-1)
pred = logits.argmax(-1)          # class indices 0–9

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
for k, (c, p) in enumerate(zip(pred.tolist(), probs.max(-1).values.tolist())):
    print(f"image {k}: {CLASSES[c]} (class {c}, {p:.1%} confidence)")
