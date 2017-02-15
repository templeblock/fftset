# fftset

fftset is designed to be an easy to use, light-weight and fast FFT library. It's designed primarily for performing fast convolutions, but can also do "usual" FFT operations. The FFTs may be vectorised (via the cop library) if the target has available vector types. The API is designed in such a way that many different sized FFTs can be created which share common memory for twiddles.

The library depends on cop (https://github.com/nickappleton/cop.git) for SIMD support which should be placed at the same level as fftset.

## Usage

See the header file for the most up-to-date usage - but for a concise summary:

> int fftset_init(struct fftset *fc);
> void fftset_destroy(struct fftset *fc);

These are used to create the structure which holds twiddles and plans for the various FFTs.

> const struct fftset_fft *fftset_create_fft(struct fftset *fc, const struct fftset_modulation *modulation, unsigned complex_bins);

Is used to create (or locate an existing) FFT object. The "modulation" parameter supports slightly different modulations from regular FFTs (for example: to provide support for different DFT variants).

> void fftset_fft_forward(const struct fftset_fft *first_pass, float *output_buf, const float *input_buf, float *work_buf);
> void fftset_fft_inverse(const struct fftset_fft *first_pass, float *output_buf, const float *input_buf, float *work_buf);

Are used to run normal FFTs.

> void fftset_fft_conv_get_kernel(const struct fftset_fft *first_pass, float *output_buf, const float *input_buf);
> void fftset_fft_conv(const struct fftset_fft *first_pass, float *output_buf, const float *input_buf, const float *kernel_buf, float *work_buf);

Are used for convolutions (which must be supported by the modulator - used to create the FFT. This information is defined in the header).

The FFT execution methods are all thread-safe (provided the work_buffers and output_buffers point to different memory locations). The FFT creation method is not thread-safe by design. Calling fftset_destroy() frees all dynamically allocated memory and causes all fftset_fft pointers to become invalid.

## Implementation

The implementation is not "complete", but is slowly moving towards a target.

Stockham DFT passes are used for the forward and inverse transforms while standard DIF and DIT transforms are used for convolution execution. This permits the vast majority of the convolution operations to be performed in place using sequential memory access patterns.

