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

#ifndef FFTSET_VEC_H
#define FFTSET_VEC_H

#include "cop/cop_alloc.h"

struct fftset_vec {
	unsigned                    lfft_div_radix;
	unsigned                    radix;
	unsigned                    cost;
	unsigned                    vec_width;

	const float                *twiddle;

	/* The best next pass to use (this pass will have:
	 *      next->lfft = this->lfft / this->radix */
	const struct fftset_vec *next_compat;

	/* If these are both null, this is the upload pass. Otherwise, these are
	 * both non-null. */
	void (*dit)(float *work_buf, unsigned nfft, unsigned lfft, const float *twid);
	void (*dif)(float *work_buf, unsigned nfft, unsigned lfft, const float *twid);
	void (*dif_stockham)(float *out, const float *in, const float *twid, unsigned ncol, unsigned nrow_div_radix);

	void (*mulconj)(float *work, const float *kern, unsigned nb_vec_fft);

	/* Position in list of all passes of this type (outer or inner pass). */
	struct fftset_vec       *next;
};

const struct fftset_vec *
fastconv_get_inner_pass
	(struct fftset_vec       **list
	,struct cop_salloc_iface  *alloc
	,unsigned                  length
	,unsigned                  vec_len
	);

void
fftset_vec_kern
	(const struct fftset_vec  *vec_pass
	,unsigned                  nb_vec_fft
	,float                    *work_buf
	);

/* The final output will be conjugated! */
void
fftset_vec_conv
	(const struct fftset_vec  *first_pass
	,unsigned                  nb_vec_fft
	,float                    *work_buf
	,const float              *kernel_buf
	);

float *
fftset_vec_stockham
	(const struct fftset_vec  *vec_pass
	,unsigned                  nb_vec_fft
	,float                    *input_buf
	,float                    *temp_buf
	);

#endif /* FFTSET_VEC_H */
