import torch
from torchvision import datasets, transforms
import os
import subprocess
import random
import sys
import time
import shutil

# ================= CONFIGURARE =================
# Calea catre Vitis Unified (Verifica sa fie corecta!)
VITIS_CMD = r"C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat"

# Setari
TCL_SCRIPT = "run_hls.tcl"
RESULT_FILENAME = "hls_result.txt"
# ===============================================

# Cai automate
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
HLS_DIR = os.path.join(PROJECT_ROOT, "hls")
PYTHON_DIR = os.path.join(PROJECT_ROOT, "python")
RESULT_FILE = os.path.join(HLS_DIR, RESULT_FILENAME)

# Import Model
sys.path.append(PYTHON_DIR)
try:
    from model import SmallCNN
except ImportError:
    print("!!! EROARE CRITICA: Nu gasesc 'model.py' !!!")
    sys.exit(1)

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

def draw_digit_ascii(img_tensor):
    """Deseneaza cifra in consola folosind caractere ASCII"""
    print(f"\n{Colors.BOLD}INPUT IMAGE VISUALIZATION (28x28):{Colors.RESET}")
    print(f"{Colors.DIM}" + "-"*30 + f"{Colors.RESET}")
    
    # Imaginea e 1x28x28, o facem 28x28
    img = img_tensor.squeeze().numpy()
    
    for row in img:
        line = ""
        for pixel in row:
            # Daca pixelul e mai intens de 0.5, punem #, altfel punct
            if pixel > 0.5:
                line += f"{Colors.BOLD}#{Colors.RESET}"
            elif pixel > 0.2:
                line += ":"
            else:
                line += " "
        print(f"|{line}|")
    
    print(f"{Colors.DIM}" + "-"*30 + f"{Colors.RESET}\n")

def generate_hls_image(img, label):
    img_data = img.numpy().flatten()
    header_path = os.path.join(HLS_DIR, "test_image.h")
    try:
        with open(header_path, "w") as f:
            f.write(f"#ifndef TEST_IMAGE_H\n#define TEST_IMAGE_H\n\n")
            f.write("const float test_image[784] = {\n")
            for i, val in enumerate(img_data):
                f.write(f"{val:.6f}, ")
                if (i + 1) % 15 == 0: f.write("\n")
            f.write("\n};\n\n")
            f.write(f"const int expected_label = {label};\n")
            f.write("#endif\n")
    except Exception as e:
        print(f"{Colors.RED}[ERR] Header gen failed: {e}{Colors.RESET}")
        sys.exit(1)

def run_hls_simulation_live():
    # 1. Stergem rezultatul vechi pentru a nu citi date expirate
    if os.path.exists(RESULT_FILE):
        try: os.remove(RESULT_FILE)
        except: pass

    cmd = [VITIS_CMD, "--mode", "hls", "--tcl", TCL_SCRIPT]
    
    print(f"[{Colors.CYAN}SYSTEM{Colors.RESET}] Starting Vitis HLS Hardware Simulation...")
    # print(f"{Colors.DIM}Command: {' '.join(cmd)}{Colors.RESET}\n") # Optional: ascundem comanda lunga

    try:
        process = subprocess.Popen(
            cmd,
            cwd=HLS_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Citim output-ul ca sa vedem progresul
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                line = line.strip()
                if "HLS PREDICTION" in line:
                    print(f"  [{Colors.GREEN}FPGA CORE{Colors.RESET}] {Colors.BOLD}{line}{Colors.RESET}")
                elif "ERROR" in line: # Doar erorile critice
                     print(f"  [{Colors.RED}LOG{Colors.RESET}] {line}")

        # Asteptam sa se inchida procesul
        process.wait()

        # === FIX-UL ESTE AICI ===
        # Intai verificam daca avem rezultatul (Fisierul hls_result.txt)
        # Daca fisierul exista si e valid, ignoram returncode-ul procesului!
        if os.path.exists(RESULT_FILE):
            with open(RESULT_FILE, "r") as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    # Succes! Returnam valorile
                    return int(lines[0].strip()), float(lines[1].strip())
        
        # Daca ajungem aici, inseamna ca nu avem fisierul de rezultat.
        # Abia acum ne plangem de returncode
        if process.returncode != 0:
            print(f"\n{Colors.RED}[FAIL] Procesul a returnat cod de eroare: {process.returncode}{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}[FAIL] Procesul a reusit, dar nu a generat hls_result.txt{Colors.RESET}")

        return -1, 0.0
            
    except Exception as e:
        print(f"{Colors.RED}[ERR] {e}{Colors.RESET}")
        return -1, 0.0

def main():
    os.system('cls' if os.name == 'nt' else 'clear') 
    print(f"{Colors.YELLOW}{Colors.BOLD}=== FPGA ACCELERATOR: REAL-TIME INFERENCE DEMO ==={Colors.RESET}")
    
    # 1. Incarcare Model
    device = "cpu"
    model = SmallCNN().to(device)
    path = os.path.join(PROJECT_ROOT, "models", "mnist_fp32.pt")
    
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"[{Colors.GREEN}OK{Colors.RESET}] Neural Network Weights Loaded.")
    else:
        print(f"[{Colors.YELLOW}WARN{Colors.RESET}] Using Random Weights.")
    model.eval()

    # 2. Selectare Imagine
    tr = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(os.path.join(PROJECT_ROOT, "data"), train=False, download=True, transform=tr)
    idx = random.randint(0, len(ds)-1)
    img, label = ds[idx]
    
    print(f"\nSelected Test Sample ID: {Colors.CYAN}#{idx}{Colors.RESET} | Expected Label: {Colors.CYAN}{label}{Colors.RESET}")
    
    # AICI ESTE PARTEA NOUA: DESENAM CIFRA
    draw_digit_ascii(img)
    
    # 3. Predictie Python
    with torch.no_grad():
        py_pred = model(img.unsqueeze(0)).argmax(dim=1).item()

    # 4. Predictie HLS Live
    generate_hls_image(img, label)
    hls_pred, hls_score = run_hls_simulation_live()

    # 5. Raport Final
    print(f"\n{Colors.BOLD}=== INFERENCE REPORT ==={Colors.RESET}")
    print(f"Software Reference (PyTorch FP32): {Colors.CYAN}{py_pred}{Colors.RESET}")
    
    if hls_pred != -1:
        status = f"{Colors.GREEN}MATCH{Colors.RESET}" if hls_pred == label else f"{Colors.RED}MISMATCH{Colors.RESET}"
        print(f"Hardware Accelerator (HLS INT8): {Colors.GREEN}{hls_pred}{Colors.RESET} (Score: {hls_score:.0f})")
        print(f"Verification Status:             {status}")
    else:
        print(f"Hardware Accelerator:            {Colors.RED}FAILED{Colors.RESET}")
    print("========================\n")

if __name__ == "__main__":
    main()