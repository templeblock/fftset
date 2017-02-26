# fftset

fftset is designed to be an easy to use, light-weight and fast FFT library. It's designed primarily for performing fast convolutions, but can also do "usual" FFT operations. The FFTs may be vectorised (via the cop library) if the target has available vector types. The API is designed in such a way that many different sized FFTs can be created which share common memory for twiddles. The algorithm is mixed-radix and supports FFT sizes comprised of arbitrary factors of 2 and 3 and zero or one factor of 5.

The library depends on cop (https://github.com/nickappleton/cop.git) for SIMD support which should be placed at the same level as fftset.

## Usage

See the header file for the most up-to-date usage - but for a concise summary:

```c++
int fftset_init(struct fftset *fc);
void fftset_destroy(struct fftset *fc);
```

The above are used to create the structure which holds twiddles and plans for the various FFTs.

```c++
const struct fftset_fft *fftset_create_fft(struct fftset *fc, const struct fftset_modulation *modulation, unsigned complex_bins);
```

The above is used to create (or locate an existing) FFT object. The "modulation" parameter supports slightly different modulations from regular FFTs (for example: to provide support for different DFT variants).

```c++
void fftset_fft_forward(const struct fftset_fft *first_pass, float *output_buf, const float *input_buf, float *work_buf);
void fftset_fft_inverse(const struct fftset_fft *first_pass, float *output_buf, const float *input_buf, float *work_buf);
```

The above are used to run normal FFTs.

```c++
void fftset_fft_conv_get_kernel(const struct fftset_fft *first_pass, float *output_buf, const float *input_buf);
void fftset_fft_conv(const struct fftset_fft *first_pass, float *output_buf, const float *input_buf, const float *kernel_buf, float *work_buf);
```

The above are used for convolutions (which must be supported by the modulator used to create the FFT. This information is defined in the header).

The FFT execution methods are all thread-safe (provided the work_buffers and output_buffers point to different memory locations). The FFT creation method is not thread-safe by design. Calling fftset_destroy() frees all dynamically allocated memory and causes all fftset_fft pointers to become invalid.

## Implementation

Modulators define the most outer passes which are responsible for descending into (and coming back out of) a vector format available on the platform when possible.

Stockham DFT passes are used for the forward and inverse transforms while standard DIF and DIT transforms are used for convolution execution. This permits the vast majority of the convolution operations to be performed in place using sequential memory access patterns.

My initial experiments when I started putting this together showed that Stockham passes were about 2/3 the cost of performing DIF style passes with explicit re-ordering steps later on. This is expected as the data is not touched as frequently, but it would be interesting to write a single reorder pass that could re-order all the bins properly in one step (i.e. have one step that has terrible memory access patterns). If this is faster than Stockham (which has worse and worse access patterns throughout the passes), I could remove about a third of the code that has been implemented. This is a TODO.

Also on the TODOs are that only one of the modulators has been heavily optimized (the FREQOFFSETREAL modulator) and only for systems with vector widths of 4 (and somewhat 8). The scalar fallback still uses sine and cosine math calls in the outer passes. The COMPLEX modulator has not had any optimization treatment at all - and probably should.

