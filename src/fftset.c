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
#include "fftset/fftset.h"
#include "cop/cop_vec.h"
#include "cop/cop_alloc.h"
#include "fftset_modulation.h"
#include "fftset_vec.h"

#ifndef V4F_EXISTS
#error this implementation requires a vector-4 type at the moment
#endif

struct fftset_fft {
	unsigned       lfft;
	unsigned       radix;

	const float   *main_twiddle;
	const float   *reord_twiddle;

	void         (*forward)      (float *out, const float *in, const float *twid, unsigned lfft_div_radix);
	void         (*forward_reord)(float *out, const float *in, const float *twid, unsigned lfft_div_radix);

	/* Conjugate the input data (as all of the vector passes are forward-DFTs)
	 * and reorder it ready for vector passes. */
	void         (*reverse_reord)(float *out, const float *in, const float *twid, unsigned lfft_div_radix);

	/* Perform the final twiddles (if necessary), conjugate the output (again,
	 * because all inner passes are forward) and reorder in the expected
	 * way. */
	void         (*reverse)      (float *out, const float *in, const float *twid, unsigned lfft_div_radix);

	/* The best next pass to use (this pass will have:
	 *      next->lfft = this->lfft / this->radix */
	const struct fftset_vec *next_compat;

	/* Position in list of all passes of this type (outer or inner pass). */
	struct fftset_fft       *next;
};

#define FASTCONV_REAL_LEN_MULTIPLE (32)
#define FASTCONV_MAX_PASSES        (24)

void
fftset_fft_conv_get_kernel
	(const struct fftset_fft    *first_pass
	,float                      *output_buf
	,const float                *input_buf
	)
{
	const struct fftset_vec *vec_pass;
	unsigned nfft = 1;

	first_pass->forward(output_buf, input_buf, first_pass->main_twiddle, first_pass->lfft);

	for (vec_pass = first_pass->next_compat; vec_pass != NULL; vec_pass = vec_pass->next_compat) {
		vec_pass->dif(output_buf, nfft, vec_pass->lfft_div_radix, vec_pass->twiddle);
		nfft *= vec_pass->radix;
	}
}

void
fftset_fft_conv
	(const struct fftset_fft *first_pass
	,float                      *output_buf
	,const float                *input_buf
	,const float                *kernel_buf
	,float                      *work_buf
	)
{
	const struct fftset_vec *pass_stack[FASTCONV_MAX_PASSES];
	const struct fftset_vec *vec_pass;
	unsigned si = 0;
	unsigned nfft = 1;
	unsigned i;

	first_pass->forward(work_buf, input_buf, first_pass->main_twiddle, first_pass->lfft);

	for (vec_pass = first_pass->next_compat; vec_pass != NULL; vec_pass = vec_pass->next_compat) {
		assert(si < FASTCONV_MAX_PASSES);
		pass_stack[si++] = vec_pass;
		vec_pass->dif(work_buf, nfft, vec_pass->lfft_div_radix, vec_pass->twiddle);
		nfft *= vec_pass->radix;
	}

	for (i = 0; i < nfft; i++) {
		v4f dr =         v4f_ld(work_buf   + 8*i+0);
		v4f di = v4f_neg(v4f_ld(work_buf  + 8*i+4));
		v4f cr =         v4f_ld(kernel_buf + 8*i+0);
		v4f ci =         v4f_ld(kernel_buf + 8*i+4);
		v4f ra = v4f_mul(dr, cr);
		v4f rb = v4f_mul(di, ci);
		v4f ia = v4f_mul(di, cr);
		v4f ib = v4f_mul(dr, ci);
		v4f ro = v4f_add(ra, rb);
		v4f io = v4f_sub(ia, ib);
		v4f_st(work_buf + 8*i + 0, ro);
		v4f_st(work_buf + 8*i + 4, io);
	}

	while (si--) {
		vec_pass = pass_stack[si];
		nfft /= vec_pass->radix;
		vec_pass->dit(work_buf, nfft, vec_pass->lfft_div_radix, vec_pass->twiddle);
	}

	assert(nfft == 1);
	first_pass->reverse(output_buf, work_buf, first_pass->main_twiddle, first_pass->lfft);
}

void
fftset_fft_forward
	(const struct fftset_fft *first_pass
	,float                      *output_buf
	,const float                *input_buf
	,float                      *work_buf
	)
{
	const struct fftset_vec *vec_pass;
	unsigned nfft = 1;

	first_pass->forward(work_buf, input_buf, first_pass->main_twiddle, first_pass->lfft);
	
	for (vec_pass = first_pass->next_compat; vec_pass != NULL; vec_pass = vec_pass->next_compat) {
		float *tmp;

		vec_pass->dif_stockham(work_buf, output_buf, vec_pass->twiddle, vec_pass->lfft_div_radix, nfft);
		nfft      *= vec_pass->radix;

		tmp        = output_buf;
		output_buf = work_buf;
		work_buf   = tmp;
	}

	first_pass->forward_reord(output_buf, work_buf, first_pass->reord_twiddle, first_pass->lfft);

	/* We have no idea which buffer is which because of the reindexing. Copy
	 * the output buffer into the work buffer. */
	memcpy(work_buf, output_buf, sizeof(float) * first_pass->lfft * 2);
}

void
fftset_fft_inverse
	(const struct fftset_fft    *first_pass
	,float                      *output_buf
	,const float                *input_buf
	,float                      *work_buf
	)
{
	const struct fftset_vec *vec_pass;
	unsigned nfft = 1;

	first_pass->reverse_reord(work_buf, input_buf, first_pass->reord_twiddle, first_pass->lfft);

	for (vec_pass = first_pass->next_compat; vec_pass != NULL; vec_pass = vec_pass->next_compat) {
		float *tmp;

		vec_pass->dif_stockham(work_buf, output_buf, vec_pass->twiddle, vec_pass->lfft_div_radix, nfft);
		nfft      *= vec_pass->radix;

		tmp        = output_buf;
		output_buf = work_buf;
		work_buf   = tmp;
	}

	first_pass->reverse(output_buf, work_buf, first_pass->main_twiddle, first_pass->lfft);

	/* We have no idea which buffer is which because of the reindexing. Copy
	 * the output buffer into the work buffer. */
	memcpy(work_buf, output_buf, sizeof(float) * first_pass->lfft * 2);
}

void fftset_init(struct fftset *fc)
{
	aalloc_init(&fc->memory, 16, 64*1024);
	fc->first_outer = NULL;
	fc->first_inner = NULL;
}

void fftset_destroy(struct fftset *fc)
{
	aalloc_free(&fc->memory);
}

const struct fftset_fft *fftset_create_fft(struct fftset *fc, const struct fftset_modulation *modulation, unsigned complex_bins)
{
	struct fftset_fft *pass;
	struct fftset_fft **ipos;

	/* Find the pass. */
	for (pass = fc->first_outer; pass != NULL; pass = pass->next) {
		if (pass->lfft == complex_bins)
			return pass;
		if (pass->lfft < complex_bins)
			break;
	}

	/* Create new outer pass and insert it into the list. */
	pass = aalloc_alloc(&fc->memory, sizeof(*pass));
	if (pass == NULL)
		return NULL;

	/* Create memory for twiddle coefficients. */
	if (modulation->get_twid != NULL) {
		pass->main_twiddle = modulation->get_twid(&fc->memory, complex_bins);
		if (pass->main_twiddle == NULL)
			return NULL;
	} else {
		pass->main_twiddle = NULL;
	}

	if (modulation->get_twid_reord != NULL) {
		pass->reord_twiddle = modulation->get_twid_reord(&fc->memory, complex_bins);
		if (pass->reord_twiddle == NULL)
			return NULL;
	} else {
		pass->reord_twiddle = NULL;
	}

	/* Create inner passes recursively. */
	pass->next_compat = fastconv_get_inner_pass(fc, complex_bins / modulation->radix);
	if (pass->next_compat == NULL)
		return NULL;

	pass->lfft          = complex_bins;
	pass->radix         = modulation->radix;
	pass->forward       = modulation->forward;
	pass->reverse       = modulation->reverse;
	pass->forward_reord = modulation->forward_reord;
	pass->reverse_reord = modulation->reverse_reord;

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
	unsigned length = 2;
	while (length < min) {
		length    *= 2;
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
	return FASTCONV_REAL_LEN_MULTIPLE * rounduptonearestfactorisation((min_real_dft_length + FASTCONV_REAL_LEN_MULTIPLE - 1) / FASTCONV_REAL_LEN_MULTIPLE) / 2;
}

