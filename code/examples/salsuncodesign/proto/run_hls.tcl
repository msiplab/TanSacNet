open_project -reset est_df_prj
set_top est_df_top
add_files est_df.cpp
open_solution -reset sol -flow_target vitis
set_part xcu250-figd2104-2L-e
create_clock -period 3.333
config_compile -pipeline_style flp
csynth_design
exit
