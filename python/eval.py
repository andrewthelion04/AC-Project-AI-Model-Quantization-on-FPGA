import argparse
import torch
from model import SmallCNN
from data import mnist_loaders

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_ds, _, test_dl = mnist_loaders()

    model = SmallCNN().to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    correct = 0
    with torch.no_grad():
        for x,y in test_dl:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
    acc = correct / len(test_ds)
    print(f"ckpt={args.ckpt} acc={acc:.4f}")

if __name__ == "__main__":
    main()
