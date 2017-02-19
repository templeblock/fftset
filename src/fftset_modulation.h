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

#include "fftset_vec.h"

struct fftset_fft;

struct fftset_modulation {
	int          (*init)(struct fftset_fft *fft, struct fftset_vec **veclist, struct cop_salloc_iface *alloc, unsigned complex_len);
};

struct fftset_fft {
	/* The following members are used by fftset in searches for particular
	 * FFTs. */
	unsigned                        lfft;
	const struct fftset_modulation *modulator;
	struct fftset_fft              *next;

	/* You are free to modify the rest of the members to suit your needs. */
	const struct fftset_vec        *next_compat;
	const float                    *main_twiddle;
	void                          (*get_kern)(const struct fftset_fft *fft, float *out, const float *in);
	void                          (*fwd)(const struct fftset_fft *fft, float *out, const float *in, float *work);
	void                          (*inv)(const struct fftset_fft *fft, float *out, const float *in, float *work);
	void                          (*conv)(const struct fftset_fft *fft, float *out, const float *in, const float *kern, float *work);
};

#endif /* FFTSET_MODULATION_H */
