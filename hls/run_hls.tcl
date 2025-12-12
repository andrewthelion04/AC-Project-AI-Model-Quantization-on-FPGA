# Vitis HLS TCL script
# Usage: In Vitis HLS: File -> Run Tcl Script -> hls/run_hls.tcl

open_project -reset qnn_dense_hls
set_top dense_int8
add_files dense_int8.cpp
add_files -tb tb_dense.cpp

# Choose a target part (edit this):
# Common choices:
#   xc7z020clg400-1   (Zynq-7000)
#   xc7a35tcsg324-1   (Artix-7)
open_solution -reset "sol1"
set_part {xc7z020clg400-1}
create_clock -period 10 -name default

csim_design
csynth_design

# Export IP if you want to import in Vivado:
export_design -format ip_catalog

exit
