add_library(fftset STATIC fftset.c fftset_mod_freqoffsetreal.c fftset_vec.c fftset_mod_cplx.c)
target_include_directories(fftset PUBLIC "../..")
