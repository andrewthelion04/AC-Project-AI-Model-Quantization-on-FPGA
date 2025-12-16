import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import mnist_loaders
from quant_utils import FakeQuantSym

# --- 1. Definitie Model Standard (FP32) ---
# (Trebuie sa fie identic cu cel folosit la antrenarea FP32)
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16*28*28, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --- 2. Definitie Model QAT (trebuie sa fie identic cu cel din qat_train.py) ---
class QATConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bits_w=8, bits_a=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k//2)
        self.fq_w = FakeQuantSym(bits_w)
        self.fq_a = FakeQuantSym(bits_a)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fq_a(x)
        w = self.fq_w(self.conv.weight)
        x = F.conv2d(x, w, self.conv.bias, stride=1, padding=self.conv.padding)
        return self.relu(x)

class QATSmallCNN(nn.Module):
    def __init__(self, bits_w=8, bits_a=8):
        super().__init__()
        self.c1 = QATConv(1, 8, 3, bits_w, bits_a)
        self.c2 = QATConv(8, 16, 3, bits_w, bits_a)
        self.fc = nn.Linear(16*28*28, 10)
        self.fq_w_fc = FakeQuantSym(bits_w)
        self.fq_a_fc = FakeQuantSym(bits_a)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = x.view(x.size(0), -1)
        x = self.fq_a_fc(x)
        w = self.fq_w_fc(self.fc.weight)
        return F.linear(x, w, self.fc.bias)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    # Am adaugat argumentele necesare pentru QAT
    parser.add_argument("--qat", action="store_true", help="Evaluate a QAT model")
    parser.add_argument("--bits", type=int, default=8, choices=[8,4], help="Bitwidth for QAT")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_ds, _, test_dl = mnist_loaders()

    # Selectam arhitectura corecta pe baza flag-ului --qat
    if args.qat:
        print(f"Initializing QAT Model (INT{args.bits})...")
        model = QATSmallCNN(bits_w=args.bits, bits_a=args.bits).to(device)
    else:
        print("Initializing Standard FP32/PTQ Model...")
        model = SmallCNN().to(device)

    # Incarcarea ponderilor
    try:
        sd = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(sd)
        print(f"Loaded checkpoint: {args.ckpt}")
    except RuntimeError as e:
        print("\nERROR: Architecture mismatch!")
        print("Daca incerci sa incarci un model QAT, nu uita sa pui flag-ul '--qat'.")
        print("Daca incerci sa incarci un model FP32, nu pune flag-ul '--qat'.\n")
        raise e

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()

    acc = correct / len(test_ds)
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()