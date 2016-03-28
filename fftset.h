/* Copyright (c) 2016 Nick Appleton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE. */

#ifndef FFTSET_H
#define FFTSET_H

#include "cop/cop_alloc.h"

/* How to use this API:
 *
 *   1) Initialize an fftset object. This will hold all the memory for the
 *      different DFT coefficients.
 *   2) Create an fft pass object. This will describe a particular DFT (or
 *      variety on a DFT) matrix depending on the constructor used. Currently
 *      there is only one constructor:
 *        - fftset_get_real_conv() which creates a real input, complex
 *          output modulation. The frequency vector is shifted by 0.5.
 *          Convolution still works on the output.
 *   3) Depending on whether you are using this DFT for convolution or ordered
 *      analysis:
 *        - Convolution: use fftset_execute_fwd() to create the frequency
 *          domain filter kernel. Then use calls to fftset_execute_conv() to
 *          perform convolution operations.
 *        - Analysis (output bins required to be in a particular order): use
 *          fftset_execute_fwd_reord() and fftset_execute_rev_reord() to
 *          perform the forward/inverse modulations respectively. */

/* Holds the memory for a variety of different DFTs. */
struct fftset;

/* Describes a particular modulation. */
struct fftset_modulation;

/* Describes how a particular modulation should be executed. */
struct fftset_fft;

/* Creates an fftset which will be used by convolution kernels. This provides
 * a way to reuse twiddles and memory so that each time a kernel is
 * constructed, there is only a minimal amount of work that needs to be done.
 */
void fftset_init(struct fftset *fc);

/* Destroys an fftset. Once this function has been called, all kernel objects
 * which were created using the given fftset must not be executed. The kernel
 * objects may be destroyed via fftset_kernel_real_destroy() after this
 * function is called. */
void fftset_destroy(struct fftset *fc);

const struct fftset_fft *fftset_create_fft(struct fftset *fc, const struct fftset_modulation *modulation, unsigned complex_bins);

void
fftset_fft_forward
	(const struct fftset_fft    *first_pass
	,const float                *input_buf
	,float                      *output_buf
	,float                      *work_buf
	);

void
fftset_fft_inverse
	(const struct fftset_fft    *first_pass
	,const float                *input_buf
	,float                      *output_buf
	,float                      *work_buf
	);

void
fftset_fft_conv_get_kernel
	(const struct fftset_fft    *first_pass
	,const float                *input_buf
	,float                      *output_buf
	);

void
fftset_fft_conv
	(const struct fftset_fft    *first_pass
	,const float                *input_buf
	,const float                *kernel_buf
	,float                      *output_buf
	,float                      *work_buf
	);

/* Available modulations
 * --------------------- */

/* FFTSET_MODULATION_FREQ_OFFSET_REAL describes a forward modulation:
 *
 *   X[k] = \sum\limits_{n=0}^{N-1} x[n] e^{ \frac{-j 2 \pi n (k + 0.5)}{N} }
 *
 * Where x[n] is a real sequence.
 *
 * This is a great modulation. I like this modulation. For N real inputs, it
 * gives N/2 complex outputs (as opposed to the normal real DFT modulation
 * which gives N/2-1 complex and 2 real outputs). It is a modulation which
 * still supports convolution. Obviously, there is no DC which makes it
 * unsuitable for some uses. */
extern const struct fftset_modulation *FFTSET_MODULATION_FREQ_OFFSET_REAL;

/* Given a particular kernel length and a maximum usage block size, give a
 * reasonably optimal length to use for the convolution FFT. The result is
 * guaranteed to be useable by fftset and will be
 * greater than kernel_length + max_block_size - 1.
 *
 * The actual maximum block size which can be used can be found by the result
 * of this function plus one minus the kernel length. */
unsigned fftset_recommend_conv_length(unsigned kernel_length, unsigned max_block_size);

/* ------------------------------------------------------------------------
 * Only defined so you can shove it on the stack. Don't access directly.
 * ------------------------------------------------------------------------ */
struct fftset_vec;

struct fftset {
	/* Sorted list of all available inner vector passes. */
	struct fftset_vec    *first_inner;
	/* Sorted list of all available outer passes. */
	struct fftset_fft    *first_outer;
	/* Memory for everything! */
	struct aalloc         memory;
};

#endif /* FFTSET_H */
