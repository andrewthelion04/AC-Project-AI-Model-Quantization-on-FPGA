# Crearea proiectului
open_project -reset proj_dense

# Setarea functiei de top (numele functiei din cpp)
set_top dense_int8

# Adaugarea fisierelor sursa
add_files dense_int8.cpp
add_files weights.h

# Adaugarea testbench-ului
add_files -tb tb_dense.cpp
add_files -tb test_image.h

# Setarea solutiei (FPGA-ul tinta - aici un Zynq generic)
open_solution -reset "solution1"
set_part {xc7z020clg400-1}
create_clock -period 10 -name default

# 1. Simulare C (Verifica logica pe PC)
csim_design

# 2. Sinteza C (Genereaza Verilog si Statistici)
csynth_design

# 3. Export RTL (Optional, daca vrei sa folosesti in Vivado)
# export_design -format ip_catalog

exit