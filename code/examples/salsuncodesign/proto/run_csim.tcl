open_project -reset est_df_csim
set_top est_df_top
add_files est_df.cpp
add_files -tb tb_est_df.cpp
add_files -tb testdata
open_solution -reset sol -flow_target vitis
set_part xcu250-figd2104-2L-e
create_clock -period 3.333
csim_design
exit
