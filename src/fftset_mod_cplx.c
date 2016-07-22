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

#include "fftset_modulation.h"
#include "cop/cop_vec.h"
#include <assert.h>
#include <math.h>

#if VLF_WIDTH != 4
#error "this implementation requires VLF_WIDTH to be 4 at the moment"
#endif

static void modcplx_forward_first(float *vec_output, const float *input, const float *coefs, unsigned fft_len)
{
	unsigned j;
	for (j = 0; j < fft_len / 4; j++) {
		float r0 = input[0*fft_len/2+2*j+0];
		float i0 = input[0*fft_len/2+2*j+1];
		float r1 = input[1*fft_len/2+2*j+0];
		float i1 = input[1*fft_len/2+2*j+1];
		float r2 = input[2*fft_len/2+2*j+0];
		float i2 = input[2*fft_len/2+2*j+1];
		float r3 = input[3*fft_len/2+2*j+0];
		float i3 = input[3*fft_len/2+2*j+1];
		
		/* 4 point complex fft */
		float yr0 = r0 + r2;
		float yi0 = i0 + i2;
		float yr2 = r0 - r2;
		float yi2 = i0 - i2;
		float yr1 = r1 + r3;
		float yi1 = i1 + i3;
		float yr3 = r1 - r3;
		float yi3 = i1 - i3;

		float tr0 = yr0 + yr1;
		float ti0 = yi0 + yi1;
		float tr2 = yr0 - yr1;
		float ti2 = yi0 - yi1;
		float tr1 = yr2 + yi3;
		float ti1 = yi2 - yr3;
		float tr3 = yr2 - yi3;
		float ti3 = yi2 + yr3;

		/* twiddles */
		float twr, twi;

		vec_output[8*j+0] = tr0;
		vec_output[8*j+4] = ti0;

		twr = cos(j * -2.0 * M_PI / (fft_len / 1));
		twi = sin(j * -2.0 * M_PI / (fft_len / 1));
		vec_output[8*j+1] = twr * tr1 - twi * ti1;
		vec_output[8*j+5] = twr * ti1 + twi * tr1;

		twr = cos(j * -4.0 * M_PI / (fft_len / 1));
		twi = sin(j * -4.0 * M_PI / (fft_len / 1));
		vec_output[8*j+2] = twr * tr2 - twi * ti2;
		vec_output[8*j+6] = twr * ti2 + twi * tr2;

		twr = cos(j * -6.0 * M_PI / (fft_len / 1));
		twi = sin(j * -6.0 * M_PI / (fft_len / 1));
		vec_output[8*j+3] = twr * tr3 - twi * ti3;
		vec_output[8*j+7] = twr * ti3 + twi * tr3;
	}
}

static void modcplx_inverse_final(float *vec_output, const float *input, const float *coefs, unsigned fft_len)
{
	unsigned j;
	for (j = 0; j < fft_len / 4; j++) {
		float r0 = input[8*j+0];
		float r1 = input[8*j+1];
		float r2 = input[8*j+2];
		float r3 = input[8*j+3];
		float i0 = input[8*j+4];
		float i1 = input[8*j+5];
		float i2 = input[8*j+6];
		float i3 = input[8*j+7];
		float twr, twi;
		float tr, ti;

		twr = cos(j * -2.0 * M_PI / (fft_len / 1));
		twi = sin(j * -2.0 * M_PI / (fft_len / 1));
		tr = r1 * twr - i1 * twi;
		ti = r1 * twi + i1 * twr;
		r1 = tr;
		i1 = ti;

		twr = cos(j * -4.0 * M_PI / (fft_len / 1));
		twi = sin(j * -4.0 * M_PI / (fft_len / 1));
		tr = r2 * twr - i2 * twi;
		ti = r2 * twi + i2 * twr;
		r2 = tr;
		i2 = ti;

		twr = cos(j * -6.0 * M_PI / (fft_len / 1));
		twi = sin(j * -6.0 * M_PI / (fft_len / 1));
		tr = r3 * twr - i3 * twi;
		ti = r3 * twi + i3 * twr;
		r3 = tr;
		i3 = ti;

		/* 4 point complex fft */
		float yr0 = r0 + r2;
		float yi0 = i0 + i2;
		float yr2 = r0 - r2;
		float yi2 = i0 - i2;
		float yr1 = r1 + r3;
		float yi1 = i1 + i3;
		float yr3 = r1 - r3;
		float yi3 = i1 - i3;

		float tr0 = yr0 + yr1;
		float ti0 = yi0 + yi1;
		float tr2 = yr0 - yr1;
		float ti2 = yi0 - yi1;
		float tr1 = yr2 + yi3;
		float ti1 = yi2 - yr3;
		float tr3 = yr2 - yi3;
		float ti3 = yi2 + yr3;

		vec_output[0*fft_len/2+2*j+0] = tr0;
		vec_output[0*fft_len/2+2*j+1] = -ti0;
		vec_output[1*fft_len/2+2*j+0] = tr1;
		vec_output[1*fft_len/2+2*j+1] = -ti1;
		vec_output[2*fft_len/2+2*j+0] = tr2;
		vec_output[2*fft_len/2+2*j+1] = -ti2;
		vec_output[3*fft_len/2+2*j+0] = tr3;
		vec_output[3*fft_len/2+2*j+1] = -ti3;
	}
}

static void modcplx_forward_reorder(float *out_buf, const float *in_buf, const float *twid, unsigned lfft)
{
	unsigned i;
	for (i = 0; i < lfft / 8; i++) {
		v4f a, b, c, d;
		V4F_LD2(a, b, in_buf + 16*i + 0);
		V4F_LD2(c, d, in_buf + 16*i + 8);
		V4F_ST2X2INT(out_buf + 16*i + 0, out_buf + 16*i + 8, a, b, c, d);
	}
}

static void modcplx_inverse_reorder(float *out_buf, const float *in_buf, const float *twid, unsigned lfft)
{
	unsigned i;
	for (i = 0; i < lfft / 8; i++) {
		v4f a, b, c, d;
		V4F_LD2X2DINT(a, b, c, d, in_buf + i*16 + 0, in_buf + i*16 + 8);
		b = v4f_neg(b);
		d = v4f_neg(d);
		V4F_ST2(out_buf + 16*i + 0, a, b);
		V4F_ST2(out_buf + 16*i + 8, c, d);
	}
}

static const struct fftset_modulation FFTSET_MODULATION_COMPLEX_DEF =
{   4
,   NULL
,   NULL
,   modcplx_forward_first
,   modcplx_forward_reorder
,   modcplx_inverse_reorder
,   modcplx_inverse_final
};

const struct fftset_modulation *FFTSET_MODULATION_COMPLEX = &FFTSET_MODULATION_COMPLEX_DEF;

