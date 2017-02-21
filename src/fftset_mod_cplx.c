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
#include "fftset_vec.h"
#include "cop/cop_vec.h"
#include <assert.h>
#include <math.h>
#include <string.h>

#if V4F_EXISTS
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

		twr = cosf(j * (float)(-2.0 * M_PI) / (fft_len / 1));
		twi = sinf(j * (float)(-2.0 * M_PI) / (fft_len / 1));
		vec_output[8*j+1] = twr * tr1 - twi * ti1;
		vec_output[8*j+5] = twr * ti1 + twi * tr1;

		twr = cosf(j * (float)(-4.0 * M_PI) / (fft_len / 1));
		twi = sinf(j * (float)(-4.0 * M_PI) / (fft_len / 1));
		vec_output[8*j+2] = twr * tr2 - twi * ti2;
		vec_output[8*j+6] = twr * ti2 + twi * tr2;

		twr = cosf(j * (float)(-6.0 * M_PI) / (fft_len / 1));
		twi = sinf(j * (float)(-6.0 * M_PI) / (fft_len / 1));
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

		twr = cosf(j * (float)(-2.0 * M_PI) / (fft_len / 1));
		twi = sinf(j * (float)(-2.0 * M_PI) / (fft_len / 1));
		tr = r1 * twr - i1 * twi;
		ti = r1 * twi + i1 * twr;
		r1 = tr;
		i1 = ti;

		twr = cosf(j * (float)(-4.0 * M_PI) / (fft_len / 1));
		twi = sinf(j * (float)(-4.0 * M_PI) / (fft_len / 1));
		tr = r2 * twr - i2 * twi;
		ti = r2 * twi + i2 * twr;
		r2 = tr;
		i2 = ti;

		twr = cosf(j * (float)(-6.0 * M_PI) / (fft_len / 1));
		twi = sinf(j * (float)(-6.0 * M_PI) / (fft_len / 1));
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

static
void
modcplx_get_kernel_v4f
	(const struct fftset_fft *first_pass
	,float                   *output_buf
	,const float             *input_buf
	)
{
	modcplx_forward_first(output_buf, input_buf, first_pass->main_twiddle, first_pass->lfft);
	fftset_vec_kern(first_pass->next_compat, 1, output_buf);
}

static
void
modcplx_conv_v4f
	(const struct fftset_fft *first_pass
	,float                   *output_buf
	,const float             *input_buf
	,const float             *kernel_buf
	,float                   *work_buf
	)
{
	modcplx_forward_first(work_buf, input_buf, first_pass->main_twiddle, first_pass->lfft);
	fftset_vec_conv(first_pass->next_compat, 1, work_buf, kernel_buf);
	modcplx_inverse_final(output_buf, work_buf, first_pass->main_twiddle, first_pass->lfft);
}

static
void
modcplx_forward_v4f
	(const struct fftset_fft *first_pass
	,float                   *output_buf
	,const float             *input_buf
	,float                   *work_buf
	)
{
	const unsigned lfft = first_pass->lfft;
	unsigned i;

	modcplx_forward_first(work_buf, input_buf, first_pass->main_twiddle, lfft);

	input_buf = fftset_vec_stockham(first_pass->next_compat, 1, work_buf, output_buf);

	if (input_buf == output_buf) {
		for (i = 0; i < lfft / 4; i++) {
			v4f a, b;
			V4F_LD2(a, b, output_buf + 8*i + 0);
			V4F_ST2INT(output_buf + 8*i, a, b);
		}
	} else {
		for (i = 0; i < lfft / 4; i++) {
			v4f a, b;
			V4F_LD2(a, b, work_buf + 8*i + 0);
			V4F_ST2INT(output_buf + 8*i, a, b);
		}
	}
}

static
void
modcplx_inverse_v4f
	(const struct fftset_fft    *first_pass
	,float                      *output_buf
	,const float                *input_buf
	,float                      *work_buf
	)
{
	const unsigned lfft = first_pass->lfft;
	unsigned i;

	for (i = 0; i < lfft / 4; i++) {
		v4f a, b;
		V4F_LD2DINT(a, b, input_buf + i*8);
		b = v4f_neg(b);
		V4F_ST2(work_buf + 8*i, a, b);
	}

	input_buf = fftset_vec_stockham(first_pass->next_compat, 1, work_buf, output_buf);

	if (input_buf != work_buf) {
		assert(input_buf == output_buf);
		memcpy(work_buf, output_buf, sizeof(float) * lfft * 2);
	}

	modcplx_inverse_final(output_buf, work_buf, first_pass->main_twiddle, lfft);
}
#endif

static
void
modcplx_get_kernel_v1f
	(const struct fftset_fft *first_pass
	,float                   *output_buf
	,const float             *input_buf
	)
{
	memcpy(output_buf, input_buf, sizeof(float) * first_pass->lfft * 2);
	fftset_vec_kern(first_pass->next_compat, 1, output_buf);
}

static
void
modcplx_conv_v1f
	(const struct fftset_fft *first_pass
	,float                   *output_buf
	,const float             *input_buf
	,const float             *kernel_buf
	,float                   *work_buf
	)
{
	const unsigned lfft = first_pass->lfft;
	unsigned i;
	memcpy(work_buf, input_buf, sizeof(float) * lfft * 2);
	fftset_vec_conv(first_pass->next_compat, 1, work_buf, kernel_buf);
	for (i = 0; i < lfft; i++) {
		output_buf[2*i+0] =  work_buf[2*i+0];
		output_buf[2*i+1] = -work_buf[2*i+1];
	}
}

static
void
modcplx_forward_v1f
	(const struct fftset_fft *first_pass
	,float                   *output_buf
	,const float             *input_buf
	,float                   *work_buf
	)
{
	const unsigned lfft = first_pass->lfft;
	memcpy(work_buf, input_buf, sizeof(float) * lfft * 2);
	input_buf = fftset_vec_stockham(first_pass->next_compat, 1, work_buf, output_buf);
	if (input_buf != output_buf)
		memcpy(output_buf, work_buf, sizeof(float) * lfft * 2);
}


static
void
modcplx_inverse_v1f
	(const struct fftset_fft    *first_pass
	,float                      *output_buf
	,const float                *input_buf
	,float                      *work_buf
	)
{
	const unsigned lfft = first_pass->lfft;
	unsigned i;
	for (i = 0; i < lfft; i++) {
		work_buf[2*i+0] =  input_buf[2*i+0];
		work_buf[2*i+1] = -input_buf[2*i+1];
	}
	input_buf = fftset_vec_stockham(first_pass->next_compat, 1, work_buf, output_buf);
	if (input_buf == output_buf) {
		for (i = 0; i < lfft; i++) {
			output_buf[2*i+0] =  output_buf[2*i+0];
			output_buf[2*i+1] = -output_buf[2*i+1];
		}
	} else {
		for (i = 0; i < lfft; i++) {
			output_buf[2*i+0] =  work_buf[2*i+0];
			output_buf[2*i+1] = -work_buf[2*i+1];
		}
	}
}

static int modcplx_init(struct fftset_fft *fft, struct fftset_vec **veclist, struct cop_salloc_iface *alloc, unsigned complex_len)
{
#if V4F_EXISTS
	if (complex_len > 4 && (complex_len % 4) == 0) {
		fft->next_compat = fastconv_get_inner_pass(veclist, alloc, complex_len / 4, 4);
		if (fft->next_compat == NULL)
			return -1;
		fft->main_twiddle = NULL;
		fft->get_kern     = modcplx_get_kernel_v4f;
		fft->fwd          = modcplx_forward_v4f;
		fft->inv          = modcplx_inverse_v4f;
		fft->conv         = modcplx_conv_v4f;
	}
	else
#endif
	{
		fft->next_compat = fastconv_get_inner_pass(veclist, alloc, complex_len, 1);
		if (fft->next_compat == NULL)
			return -1;
		fft->main_twiddle = NULL;
		fft->get_kern     = modcplx_get_kernel_v1f;
		fft->fwd          = modcplx_forward_v1f;
		fft->inv          = modcplx_inverse_v1f;
		fft->conv         = modcplx_conv_v1f;
	}

	return 0;
}

static const struct fftset_modulation FFTSET_MODULATION_COMPLEX_DEF =
{   modcplx_init
};

const struct fftset_modulation *FFTSET_MODULATION_COMPLEX = &FFTSET_MODULATION_COMPLEX_DEF;

