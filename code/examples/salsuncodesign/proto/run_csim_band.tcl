set cfl "-DSZY=$::env(SZY) -DSZX=$::env(SZX) -DBAND=$::env(BAND) -DHALO=$::env(HALO)"
open_project -reset band_df_csim
set_top salsun2d_band_df
add_files band_df.cpp -cflags $cfl
add_files est_df.cpp
add_files -tb tb_band_df.cpp -cflags $cfl
add_files -tb testdata_band
open_solution -reset sol -flow_target vitis
set_part xcu250-figd2104-2L-e
create_clock -period 3.333
csim_design
exit
