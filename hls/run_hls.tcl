# 1. Creare Proiect
open_project -reset proj_mnist_hls

# 2. Adaugare fisiere sursa (Codul C++ al acceleratorului)
add_files dense_int8.cpp
add_files -tb tb_dense.cpp  ; # Testbench-ul
add_files -tb test_image.h  ; # Imaginea generata de Python

# 3. Setare Top Function (Functia principala)
set_top dense_int8

# 4. Configurare Solutie (Ce FPGA folosim)
open_solution -reset "solution1"
# Poti schimba part-ul cu cel de pe placa ta (ex: PYNQ-Z2, Zybo)
# Aici e un part generic Zynq 7020
set_part {xc7z020clg400-1}
create_clock -period 10 -name default

# ================= ETAPELE DE EXECUTIE =================

# A. C Simulation (Verifica corectitudinea logica)
# Asta faceai tu pana acum cand rulai demo.py
csim_design

# B. C Synthesis (Genereaza Rapoartele LUT/FF/DSP/Latency)
# Asta va dura cateva minute!
csynth_design

# C. Co-Simulation (Optional - Verifica RTL-ul generat)
# cosim_design ; # Dureaza mult, il tinem comentat pentru demo

# D. Export IP (Genereaza IP-ul pentru Vivado si rapoarte finale)
# Aceasta comanda iti va da si estimari mai bune de resurse
export_design -format ip_catalog

# Inchide proiectul
exit