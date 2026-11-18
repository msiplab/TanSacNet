# EST_DEFS: estimator options of est_df.h (e.g. "-DEST_NH0=68 -DEST_NH1=36")
set est [expr {[info exists ::env(EST_DEFS)] ? $::env(EST_DEFS) : ""}]
set cfl "-DSZY=$::env(SZY) -DSZX=$::env(SZX) -DBAND=$::env(BAND) -DHALO=$::env(HALO) $est"
open_project -reset $::env(PRJ)
set_top salsun2d_band_df
add_files band_df.cpp -cflags $cfl
add_files est_df.cpp -cflags $est
open_solution -reset sol -flow_target vitis
set_part xcu250-figd2104-2L-e
create_clock -period 3.333
config_compile -pipeline_style flp
csynth_design
exit
