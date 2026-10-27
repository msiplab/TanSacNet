# Estimator cosim with the v++ pipeline style (frp) and random stalls
# on the streams and AXI ports (debugging card-only errors)
open_project -reset est_cosim
set_top est_kernel
add_files est_df.cpp
add_files est_kernel.cpp
add_files -tb tb_est_kernel.cpp
add_files -tb testdata
open_solution -reset sol -flow_target vitis
set_part xcu250-figd2104-2L-e
create_clock -period 3.333
csynth_design
cosim_design -rtl verilog -random_stall
exit
