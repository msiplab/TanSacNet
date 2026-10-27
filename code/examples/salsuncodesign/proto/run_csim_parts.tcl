open_project -reset parts_csim
set_top part_ln
add_files est_parts.cpp
add_files -tb tb_parts.cpp
add_files -tb testdata
open_solution -reset sol -flow_target vitis
set_part xcu250-figd2104-2L-e
create_clock -period 3.333
csim_design
exit
