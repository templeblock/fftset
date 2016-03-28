add_library(fftset STATIC fftset.c fftset_mod_freqoffsetreal.c fftset_vec.c)
target_include_directories(fftset PUBLIC "../..")
