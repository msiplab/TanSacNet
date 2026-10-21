set cfl "-DSZY=$::env(SZY) -DSZX=$::env(SZX) -DBAND=$::env(BAND) -DHALO=$::env(HALO)"
open_project -reset $::env(PRJ)
set_top salsun2d_band_df
add_files band_df.cpp -cflags $cfl
add_files est_df.cpp
open_solution -reset sol -flow_target vitis
set_part xcu250-figd2104-2L-e
create_clock -period 3.333
config_compile -pipeline_style flp
csynth_design
exit
