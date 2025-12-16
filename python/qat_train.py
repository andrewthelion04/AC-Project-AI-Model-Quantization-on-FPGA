import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import mnist_loaders
from quant_utils import FakeQuantSym

# --- Definitii Clase ---
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
        # Structura trebuie sa fie compatibila cu SmallCNN original
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

def load_fp32_weights(qat_model, fp32_ckpt_path, device):
    """
    Incarca ponderile dintr-un model FP32 standard (conv1, conv2)
    in structura QAT (c1.conv, c2.conv).
    """
    print(f"Loading FP32 weights from {fp32_ckpt_path}...")
    fp32_sd = torch.load(fp32_ckpt_path, map_location=device)
    qat_sd = qat_model.state_dict()

    # Harta de mapare: Cheie FP32 -> Cheie QAT
    # Ajusteaza numele daca modelul tau FP32 are alte nume
    mapping = {
        "conv1.weight": "c1.conv.weight",
        "conv1.bias":   "c1.conv.bias",
        "conv2.weight": "c2.conv.weight",
        "conv2.bias":   "c2.conv.bias",
        "fc.weight":    "fc.weight",
        "fc.bias":      "fc.bias"
    }

    with torch.no_grad():
        for fp32_key, qat_key in mapping.items():
            if fp32_key in fp32_sd:
                qat_sd[qat_key].copy_(fp32_sd[fp32_key])
            else:
                print(f"Warning: {fp32_key} not found in checkpoint!")
    
    # Incarcam inapoi in model
    qat_model.load_state_dict(qat_sd)
    print("Weights loaded successfully!")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8, choices=[8,4])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", type=str, default="models/mnist_qat_int8.pt")
    # Am adaugat acest argument care lipsea:
    ap.add_argument("--in_ckpt", type=str, default=None, help="Path to FP32 checkpoint for init")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_ds, train_dl, test_dl = mnist_loaders()

    model = QATSmallCNN(bits_w=args.bits, bits_a=args.bits).to(device)

    # Aici intervine logica noua: Daca avem checkpoint, il incarcam
    if args.in_ckpt:
        load_fp32_weights(model, args.in_ckpt, device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4) # LR mai mic pentru Fine-Tuning
    loss_fn = nn.CrossEntropyLoss()

    print(f"Starting QAT Training (Bits={args.bits})...")
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
        print(f"[QAT INT{args.bits}] epoch={epoch} acc={acc:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print("saved:", args.out)

if __name__ == "__main__":
    main()