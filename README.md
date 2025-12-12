# qnn-fpga (Proiect: Cuantizarea modelelor AI pe FPGA)

Repo "starter kit" pentru proiectul vostru (fără placă fizică), aliniat cu documentația tehnică:
- Python (PyTorch): baseline FP32 + PTQ/QAT (INT8/INT4) + export weights
- C++: fixed-point utils (saturare + requantizer Gemmlowp-style)
- HLS (Vitis HLS): kernel Dense INT8 + Requantizer, sintetizabil; rapoarte pentru Vivado

> Scop: rulați training/quant pe Mac, iar sinteza HLS/Vivado pe Windows.
> Fără placă: folosiți un "target part" în Vivado pentru estimări de resurse/timing.

## 0) Setup (Mac/Windows cu Python)
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

## 1) Baseline FP32 (MNIST)
```bash
python python/train_fp32.py
python python/eval.py --ckpt models/mnist_fp32.pt
```

## 2) PTQ (INT8/INT4) + evaluare
```bash
python python/ptq_quantize.py --bits 8 --in_ckpt models/mnist_fp32.pt --out_ckpt models/mnist_ptq_int8.pt
python python/eval.py --ckpt models/mnist_ptq_int8.pt

python python/ptq_quantize.py --bits 4 --in_ckpt models/mnist_fp32.pt --out_ckpt models/mnist_ptq_int4.pt
python python/eval.py --ckpt models/mnist_ptq_int4.pt
```

## 3) QAT (INT8/INT4) + evaluare
```bash
python python/qat_train.py --bits 8 --epochs 3 --out models/mnist_qat_int8.pt
python python/eval.py --ckpt models/mnist_qat_int8.pt

python python/qat_train.py --bits 4 --epochs 3 --out models/mnist_qat_int4.pt
python python/eval.py --ckpt models/mnist_qat_int4.pt
```

## 4) Export weights (exemplu: conv1 weights) pentru hardware
```bash
python python/export_quant.py --bits 8 --ckpt models/mnist_fp32.pt --out_dir models/export
```

## 5) C++ fixed-point (Mac)
```bash
c++ -O2 -std=c++17 cpp_fixed/fixed_point_test.cpp -o /tmp/fixed_test
/tmp/fixed_test
```

## 6) Vitis HLS + Vivado (Windows) — fără placă
### Alegeți un target FPGA "part" (exemple comune):
- xc7z020clg400-1 (Zynq-7000)
- xc7a35tcsg324-1 (Artix-7)

### Rulare Vitis HLS (prin TCL)
1) Deschideți **Vitis HLS** (sau HLS din Vitis).
2) File -> Run Tcl Script -> `hls/run_hls.tcl`
3) În TCL, editați `set_part` cu part-ul ales.

Veți obține rapoarte de: LUT/FF/DSP/BRAM + latency/II.
Apoi puteți exporta IP și îl puteți importa în Vivado pentru rapoarte suplimentare.

## Output-uri recomandate pentru predare
- tabele accuracy (FP32 vs INT8 vs INT4; PTQ vs QAT)
- rapoarte HLS: latency, initiation interval (II), utilizare resurse
- rapoarte Vivado post-synthesis: LUT/FF/DSP/BRAM, Fmax estimat
