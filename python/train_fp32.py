import os, argparse
import torch
import torch.nn as nn
from model import SmallCNN
from data import mnist_loaders

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", type=str, default="models/mnist_fp32.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_ds, train_dl, test_dl = mnist_loaders()

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        for x,y in train_dl:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x,y in test_dl:
                x,y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
        acc = correct / len(test_ds)
        print(f"[FP32] epoch={epoch} acc={acc:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print("saved:", args.out)

if __name__ == "__main__":
    main()
