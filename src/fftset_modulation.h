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

#ifndef FFTSET_MODULATION_H
#define FFTSET_MODULATION_H

#include "cop/cop_alloc.h"

struct fftset_modulation {
	unsigned       radix;

	const float *(*get_twid_reord)(struct aalloc *alloc, unsigned lfft_div_radix);
	const float *(*get_twid)(struct aalloc *alloc, unsigned lfft_div_radix);

	void         (*forward)      (float *out, const float *in, const float *twid, unsigned lfft_div_radix);
	void         (*forward_reord)(float *out, const float *in, const float *twid, unsigned lfft_div_radix);
	void         (*reverse_reord)(float *out, const float *in, const float *twid, unsigned lfft_div_radix);
	void         (*reverse)      (float *out, const float *in, const float *twid, unsigned lfft_div_radix);
};

#endif /* FFTSET_MODULATION_H */