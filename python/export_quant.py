import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np
from quant_utils import FakeQuantSym

# --- DEFINITIA MODELULUI QAT (Aceeasi ca in qat_train.py) ---
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
        pass # Nu avem nevoie de forward pentru export

def export_tensor_to_h(f, name, tensor, quant_scale=1.0, is_int=False):
    """
    Scrie un tensor intr-un fisier .h sub forma de array C++.
    Daca is_int=True, valorile sunt scrise ca intregi.
    Altfel sunt float.
    """
    data = tensor.detach().cpu().numpy().flatten()
    
    # Daca avem o scala, cuantizam valorile: Int = Float / Scale
    if quant_scale != 1.0 and not is_int:
        data = data / quant_scale
        data = np.round(data) # Rotunjire la cel mai apropiat intreg
    
    data = data.astype(int) if is_int or quant_scale != 1.0 else data
    
    f.write(f"// {name} (Shape: {tensor.shape})\n")
    
    # Alegem tipul de date pentru C++
    if name.endswith("_w"): # Weights (8-bit)
        ctype = "const int8_t"
    elif name.endswith("_b"): # Bias (32-bit de obicei pentru acumulare)
        ctype = "const int32_t"
    else:
        ctype = "const float"

    f.write(f"{ctype} {name}[] = {{\n")
    
    for i, val in enumerate(data):
        f.write(f"{val}, ")
        if (i + 1) % 10 == 0:
            f.write("\n")
    f.write("};\n\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to QAT checkpoint")
    parser.add_argument("--out_dir", type=str, default="models/export", help="Output directory")
    parser.add_argument("--bits", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cpu"

    print(f"Loading QAT Model (INT{args.bits}) from {args.ckpt}...")
    model = QATSmallCNN(bits_w=args.bits, bits_a=args.bits).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    
    # Fisierul de iesire
    header_path = os.path.join(args.out_dir, "weights.h")
    print(f"Exporting to {header_path}...")

    with open(header_path, "w") as f:
        f.write("#pragma once\n")
        f.write("#include <cstdint>\n\n")

        # --- Layer 1 (Conv) ---
        # Scale-ul greutatilor se afla in fq_w.scale
        scale_w1 = model.c1.fq_w.scale.item()
        scale_a1 = model.c1.fq_a.scale.item() # Scale-ul inputului
        
        # Export Weights (INT8)
        export_tensor_to_h(f, "conv1_w", model.c1.conv.weight, quant_scale=scale_w1)
        
        # Export Bias (De obicei bias-ul se cuantizeaza cu scale_input * scale_weights)
        # Aici il vom exporta ca INT32 pentru a fi safe in FPGA
        scale_bias1 = scale_a1 * scale_w1
        export_tensor_to_h(f, "conv1_b", model.c1.conv.bias, quant_scale=scale_bias1)

        # --- Layer 2 (Conv) ---
        scale_w2 = model.c2.fq_w.scale.item()
        # Scale-ul inputului layer 2 vine teoretic din activarea layer 1
        # Pentru simplitate, folosim fq_a intern
        scale_a2 = model.c2.fq_a.scale.item()
        
        export_tensor_to_h(f, "conv2_w", model.c2.conv.weight, quant_scale=scale_w2)
        scale_bias2 = scale_a2 * scale_w2
        export_tensor_to_h(f, "conv2_b", model.c2.conv.bias, quant_scale=scale_bias2)

        # --- Layer 3 (Fully Connected) ---
        scale_w_fc = model.fq_w_fc.scale.item()
        scale_a_fc = model.fq_a_fc.scale.item()

        export_tensor_to_h(f, "fc_w", model.fc.weight, quant_scale=scale_w_fc)
        scale_bias_fc = scale_a_fc * scale_w_fc
        export_tensor_to_h(f, "fc_b", model.fc.bias, quant_scale=scale_bias_fc)
        
        # Scriem si scalele ca sa le stim in C++ daca e nevoie de requantizare
        f.write(f"// Quantization Scales (Float references)\n")
        f.write(f"const float scale_w1 = {scale_w1};\n")
        f.write(f"const float scale_a1 = {scale_a1};\n")
        f.write(f"const float scale_w2 = {scale_w2};\n")
        f.write(f"const float scale_a2 = {scale_a2};\n")

    print("Export done!")

if __name__ == "__main__":
    main()