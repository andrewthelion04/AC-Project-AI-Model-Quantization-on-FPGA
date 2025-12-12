"""Simple PTQ: quantize weights to INT{bits} and dequantize back for inference.
This keeps the same FP32 graph but simulates quantized weights (good for accuracy trade-off study).
"""
import os, argparse
import torch
from model import SmallCNN
from quant_utils import quant_dequant_symmetric

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8, choices=[8,4])
    ap.add_argument("--in_ckpt", required=True)
    ap.add_argument("--out_ckpt", required=True)
    args = ap.parse_args()

    model = SmallCNN()
    model.load_state_dict(torch.load(args.in_ckpt, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        for name, p in model.named_parameters():
            if "weight" in name:
                p.copy_(quant_dequant_symmetric(p, args.bits)[0])

    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    torch.save(model.state_dict(), args.out_ckpt)
    print("saved:", args.out_ckpt)

if __name__ == "__main__":
    main()
