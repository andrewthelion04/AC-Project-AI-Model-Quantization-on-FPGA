"""Export a layer's weights quantized to INT{bits} + the scale.
For a full accelerator you'd export all layers; for the project, exporting conv1 is often enough.
"""
import os, csv, argparse
import torch
from model import SmallCNN
from quant_utils import calc_qparams_symmetric

def quantize_tensor_to_int(x: torch.Tensor, bits: int, scale: float):
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    q = torch.round(x / scale).clamp(qmin, qmax).to(torch.int32)
    return q

def save_csv_int(path, tensor_int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat = tensor_int.reshape(-1).tolist()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for v in flat:
            w.writerow([int(v)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8, choices=[8,4])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="models/export")
    args = ap.parse_args()

    model = SmallCNN()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    W = model.conv1.weight.data.clone()
    scale, _ = calc_qparams_symmetric(W, args.bits)
    Wq = quantize_tensor_to_int(W, args.bits, scale.item())

    save_csv_int(os.path.join(args.out_dir, f"conv1_w_int{args.bits}.csv"), Wq)
    with open(os.path.join(args.out_dir, f"conv1_w_scale_int{args.bits}.txt"), "w") as f:
        f.write(str(scale.item()))

    print("export ok:", args.out_dir)

if __name__ == "__main__":
    main()
