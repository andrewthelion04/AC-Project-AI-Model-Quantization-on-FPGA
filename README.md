# FPGA Accelerated MNIST Classifier (INT8 Quantization)

Acest proiect implementează un accelerator hardware pentru recunoașterea cifrelor (MNIST) pe FPGA, folosind o arhitectură **Hardware-Software Co-design**.

Proiectul demonstrează eficiența cuantizării **Post-Training Quantization (PTQ)**, realizând tranziția de la modele software în virgulă mobilă (FP32) la inferență hardware optimizată pe numere întregi (**INT8**). Această abordare reduce semnificativ utilizarea resurselor de memorie și procesare, fără a compromite acuratețea predicției.

## 🌟 Funcționalități Cheie

* **Pipeline Complet:** Antrenare (PyTorch) $\rightarrow$ Export $\rightarrow$ Sinteză HLS $\rightarrow$ Simulare Hardware.
* **INT8 Inference:** Acceleratorul folosește exclusiv aritmetică pe 8 biți, optimizând utilizarea blocurilor DSP și eliminând calculele costisitoare în virgulă mobilă.
* **Secvențiere (Sequence Prediction):** Demo interactiv care construiește numere complexe (ex: "592") din cifre individuale procesate secvențial de FPGA.
* **Automated Benchmarking:** Scripturi pentru compararea automată a preciziei FP32 vs. INT8 și generarea de tabele compatibile LaTeX.

---

## 🛠️ Cerințe de Sistem

* **Sistem de Operare:** Windows 10/11 (Necesar pentru Vitis HLS).
* **Software FPGA:** Xilinx Vitis HLS 2023.x (sau versiuni compatibile).
* **Limbaje & Mediu:**
    * Python 3.8+
    * C++14/17 (pentru HLS)
* **Librării Python:** `torch`, `torchvision`, `numpy` (vezi `requirements.txt`).

---

## 🚀 Configurare și Instalare

### 1. Setup Mediu Virtual

```bash
# Creare mediu virtual
python -m venv .venv

# Activare pe Windows (PowerShell):
.venv\Scripts\activate 

# Activare pe Linux/Mac:
source .venv/bin/activate

# Instalare dependințe:
pip install -r requirements.txt
```

### 2. Configurare Căi Vitis (CRITIC!)
Editați fișierul `python/demo.py` și `python/benchmark_accuracy.py` pentru a seta calea corectă către executabilul Vitis HLS de pe mașina dumneavoastră:

```python
# Exemplu de modificare în demo.py:
VITIS_CMD = r"C:\Xilinx\Vitis_HLS\2023.2\bin\vitis_hls.bat" 
```

---

## 📊 Fluxul de Lucru (Workflow)

### Pasul 1: Antrenare Model (Baseline FP32)
Antrenează rețeaua neuronală (arhitectură CNN simplificată) în PyTorch folosind precizie maximă (Floating Point 32-bit).

```bash
python python/train_fp32.py
# Output: models/mnist_fp32.pt
```

### Pasul 2: Evaluare și Benchmark (FP32 vs INT8)
Generează automat tabele de acuratețe comparând modelul software cu simularea hardware bit-exactă. Acest script generează automat și codul LaTeX necesar pentru documentația tehnică.

```bash
python python/benchmark_accuracy.py
```
*Output așteptat:* Tabel comparativ (ex: FP32: 98.50% vs INT8: 98.15%).

### Pasul 3: Sinteză Hardware și Raportare Resurse (Vitis HLS)
Acest pas transformă codul C++ (`dense_int8.cpp`) în RTL (Verilog), rulând simularea C (`csim`), sinteza (`csynth`) și exportul IP-ului.

```bash
# Se poate rula manual din consolă:
vitis_hls -f hls/run_hls.tcl
```

**Unde găsesc rapoartele?**
* **Locație:** `hls/proj_mnist_hls/solution1/syn/report/`
* **Ce informații conțin:**
    * **Latency (Cycles):** Viteza de execuție a unei predicții.
    * **Initiation Interval (II):** Throughput-ul acceleratorului.
    * **Utilization:** Consumul de resurse FPGA (DSP48E, LUT, FF, BRAM).

### Pasul 4: Demo Interactiv (Live Inference)
Scriptul principal care integrează totul. Acesta trimite imagini din setul de testare către simulatorul FPGA și afișează rezultatul în timp real.
* **Mod:** Secvențial (prezice 3 cifre consecutive pentru a forma un număr mare).
* **Vizualizare:** ASCII Art în consolă.

```bash
python python/demo.py
```

---

## 📂 Structura Proiectului

```text
.
├── hls/
│   ├── dense_int8.cpp       # Sursa C++ a acceleratorului (INT8 Core)
│   ├── tb_dense.cpp         # Testbench pentru verificare C++
│   ├── run_hls.tcl          # Script de automatizare Vitis (CSim/CSynth/Export)
│   └── proj_mnist_hls/      # (Generat) Rapoartele de sinteză și log-uri
├── models/
│   └── mnist_fp32.pt        # Greutățile modelului antrenat (salvate aici)
├── python/
│   ├── model.py             # Definiția arhitecturii CNN în PyTorch
│   ├── train_fp32.py        # Script antrenare
│   ├── demo.py              # Aplicația principală (Python <-> Vitis Bridge)
│   └── benchmark_accuracy.py# Script generare tabele precizie
├── .gitignore               # Exclude fișiere temporare și log-uri mari
└── README.md                # Documentația proiectului
```

---

## 📝 Note Tehnice pentru Documentație

1.  **De ce INT8?**
    Proiectul demonstrează că pentru clasificarea imaginilor (MNIST), precizia FP32 nu este necesară la inferență. Folosind INT8, reducem memoria necesară pentru greutăți de ~4x și utilizăm blocuri DSP optimizate pentru înmulțiri întregi, crescând throughput-ul și reducând consumul energetic.

2.  **Scoruri vs Probabilități:**
    Acceleratorul Hardware (HLS) returnează **scoruri brute (logits)**, nu probabilități (Softmax). Deoarece funcția Softmax este monotonă, valoarea maximă indică clasa corectă fără a fi nevoie de calculul complex al exponențialelor pe FPGA, economisind resurse logice semnificative.

3.  **Deadlock Prevention:**
    Interfața Python-Vitis din `demo.py` implementează `subprocess.communicate()` pentru a gestiona corect fluxurile de date (pipes), prevenind blocarea buffer-ului de ieșire (deadlock) în timpul simulărilor intensive generate de Vitis HLS.