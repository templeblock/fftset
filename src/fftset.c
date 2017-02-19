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

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "fftset/fftset.h"
#include "cop/cop_vec.h"
#include "cop/cop_alloc.h"
#include "fftset_modulation.h"
#include "fftset_vec.h"

/* To be removed when the vector FFT defines the correct function for
 * performing convolution - currently this code uses a v4f for the
 * inner FFT multiplication which will be wrong when AVX inner passes start
 * happening. */
#ifndef V4F_EXISTS
#error "this implementation requires a v4f type"
#endif

#define FASTCONV_REAL_LEN_MULTIPLE (32)

void
fftset_fft_conv_get_kernel
	(const struct fftset_fft    *first_pass
	,float                      *output_buf
	,const float                *input_buf
	)
{
	first_pass->modulator->get_kern(first_pass, output_buf, input_buf);
}

void
fftset_fft_conv
	(const struct fftset_fft    *first_pass
	,float                      *output_buf
	,const float                *input_buf
	,const float                *kernel_buf
	,float                      *work_buf
	)
{
	first_pass->modulator->conv(first_pass, output_buf, input_buf, kernel_buf, work_buf);
}

void
fftset_fft_forward
	(const struct fftset_fft *first_pass
	,float                      *output_buf
	,const float                *input_buf
	,float                      *work_buf
	)
{
	first_pass->modulator->fwd(first_pass, output_buf, input_buf, work_buf);
}

void
fftset_fft_inverse
	(const struct fftset_fft    *first_pass
	,float                      *output_buf
	,const float                *input_buf
	,float                      *work_buf
	)
{
	first_pass->modulator->inv(first_pass, output_buf, input_buf, work_buf);
}

int fftset_init(struct fftset *fc)
{
	fc->first_outer = NULL;
	fc->first_inner = NULL;
	return cop_alloc_grp_temps_init(&(fc->mem_impl), &(fc->mem), 8*1024*1024, 0, 16);
}

void fftset_destroy(struct fftset *fc)
{
#if 0
	struct fftset_vec *pass;
	for (pass = fc->first_inner; pass != NULL; pass = pass->next)
		printf("len=%u; cost=%u; radix=%u\n", pass->lfft_div_radix*pass->radix, pass->cost, pass->radix);
#endif
	cop_alloc_grp_temps_free(&(fc->mem_impl));
}

const struct fftset_fft *fftset_create_fft(struct fftset *fc, const struct fftset_modulation *modulation, unsigned complex_bins)
{
	struct fftset_fft *pass;
	struct fftset_fft **ipos;

	/* Find the pass. */
	for (pass = fc->first_outer; pass != NULL; pass = pass->next) {
		if (pass->lfft == complex_bins && pass->modulator == modulation)
			return pass;
		if (pass->lfft < complex_bins)
			break;
	}

	/* Create new outer pass and insert it into the list. */
	pass = cop_salloc(&(fc->mem), sizeof(*pass), 0);
	if (pass == NULL)
		return NULL;

	/* Create memory for twiddle coefficients. */
	if (modulation->get_twid != NULL) {
		pass->main_twiddle = modulation->get_twid(&(fc->mem), complex_bins);
		if (pass->main_twiddle == NULL)
			return NULL;
	} else {
		pass->main_twiddle = NULL;
	}

	/* Create inner passes recursively. */
	pass->next_compat = fastconv_get_inner_pass(fc, complex_bins / modulation->radix);
	if (pass->next_compat == NULL)
		return NULL;

	pass->lfft          = complex_bins;
	pass->modulator     = modulation;

	/* Insert into list. */
	ipos = &(fc->first_outer);
	while (*ipos != NULL && complex_bins < (*ipos)->lfft) {
		ipos = &(*ipos)->next;
	}
	pass->next = *ipos;
	*ipos = pass;
	return pass;
}

static unsigned rounduptonearestfactorisation(unsigned min)
{
	unsigned length = 1;
	while (length < min) {
		if (length * 2 >= min)
			length *= 2;
		else if (length * 3 >= min)
			length *= 3;
		else
			length *= 4;
	}
	return length;
}

unsigned fftset_recommend_conv_length(unsigned kernel_length, unsigned max_block_size)
{
	/* We found that input blocks of 8 times the kernel length is a good
	 * performance point. */
	const unsigned target_max_block_size = 8 * kernel_length;

	/* If the user specifies they will never pass a block larger than
	 * max_block_size and our target is less than this value, use the
	 * maximum of the two. Recall, if a user specifies a max_block_size of
	 * zero this indicates that they will deal with whatever maximum is
	 * returned by fastconv_kernel_max_block_size() after initialization
	 * (which is likely to be close to our target value). */
	const unsigned real_max_block_size = (max_block_size > target_max_block_size) ? max_block_size : target_max_block_size;

	/* The minimum real dft length is just the length of the maximum convolved
	 * output sequence. */
	const unsigned min_real_dft_length = kernel_length + real_max_block_size - 1;

	/* We do our processing on vectors if we can. The optimal way of doing
	 * this (which avoids any scalar loads/stores on input/output vectors)
	 * requires particular length multiples. */
	return (FASTCONV_REAL_LEN_MULTIPLE/2) * rounduptonearestfactorisation((min_real_dft_length + FASTCONV_REAL_LEN_MULTIPLE - 1) / FASTCONV_REAL_LEN_MULTIPLE);
}

