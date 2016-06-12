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

#if !defined(V4F_EXISTS)
/* What is the deal with this?
 * At the moment, all of the vector FFT modulators assume a vector length of
 * four. At some point, this will be changed to be the longest vector type
 * available on the system (so we can squeeze a bit more out of AVX and also
 * build the modulators on systems with no vector support...) - at this point
 * all of these modulators will also need to be updated. But for the moment,
 * because the internal FFT passes are V4, the outer FFT passes must all
 * produce V4 data. */
#error "this module requires a v4f type at this point in time"
#endif

static void modfreqoffsetreal_forward_first(float *vec_output, const float *input, const float *coefs, unsigned fft_len)
{
	const unsigned fft_len_4 = fft_len / 4;
	unsigned i;
	assert((fft_len % 16) == 0);
	for (i = fft_len / 16
		;i
		;i--, coefs += 56, vec_output += 32, input += 4) {
		v4f r1    = v4f_ld(input + 0*fft_len_4);
		v4f r2    = v4f_ld(input + 1*fft_len_4);
		v4f r3    = v4f_ld(input + 2*fft_len_4);
		v4f r4    = v4f_ld(input + 3*fft_len_4);
		v4f i1    = v4f_ld(input + 4*fft_len_4);
		v4f i2    = v4f_ld(input + 5*fft_len_4);
		v4f i3    = v4f_ld(input + 6*fft_len_4);
		v4f i4    = v4f_ld(input + 7*fft_len_4);

		v4f twr1  = v4f_ld(coefs + 0);
		v4f twi1  = v4f_ld(coefs + 4);
		v4f twr2  = v4f_ld(coefs + 8);
		v4f twi2  = v4f_ld(coefs + 12);
		v4f or1a  = v4f_mul(twr1, r1);
		v4f or1b  = v4f_mul(twi1, i1);
		v4f oi1a  = v4f_mul(twi1, r1);
		v4f oi1b  = v4f_mul(twr1, i1);
		v4f or2a  = v4f_mul(twr2, r2);
		v4f or2b  = v4f_mul(twi2, i2);
		v4f oi2a  = v4f_mul(twi2, r2);
		v4f oi2b  = v4f_mul(twr2, i2);
		v4f or1   = v4f_add(or1a, or1b);
		v4f oi1   = v4f_sub(oi1a, oi1b);
		v4f or2   = v4f_add(or2a, or2b);
		v4f oi2   = v4f_sub(oi2a, oi2b);
		v4f twr3  = v4f_ld(coefs + 16);
		v4f twi3  = v4f_ld(coefs + 20);
		v4f twr4  = v4f_ld(coefs + 24);
		v4f twi4  = v4f_ld(coefs + 28);
		v4f or3a  = v4f_mul(twr3, r3);
		v4f or3b  = v4f_mul(twi3, i3);
		v4f oi3a  = v4f_mul(twi3, r3);
		v4f oi3b  = v4f_mul(twr3, i3);
		v4f or4a  = v4f_mul(twr4, r4);
		v4f or4b  = v4f_mul(twi4, i4);
		v4f oi4a  = v4f_mul(twi4, r4);
		v4f oi4b  = v4f_mul(twr4, i4);
		v4f or3   = v4f_add(or3a, or3b);
		v4f oi3   = v4f_sub(oi3a, oi3b);
		v4f or4   = v4f_add(or4a, or4b);
		v4f oi4   = v4f_sub(oi4a, oi4b);

		v4f t0ra  = v4f_add(or1, or3);
		v4f t0rs  = v4f_sub(or1, or3);
		v4f t1ra  = v4f_add(or2, or4);
		v4f t1rs  = v4f_sub(or2, or4);
		v4f t1is  = v4f_sub(oi2, oi4);
		v4f t1ia  = v4f_add(oi2, oi4);
		v4f t0is  = v4f_sub(oi1, oi3);
		v4f t0ia  = v4f_add(oi1, oi3);
		v4f mor0  = v4f_add(t0ra, t1ra);
		v4f mor2  = v4f_sub(t0ra, t1ra);
		v4f mor1  = v4f_add(t0rs, t1is);
		v4f mor3  = v4f_sub(t0rs, t1is);
		v4f moi1  = v4f_sub(t0is, t1rs);
		v4f moi3  = v4f_add(t0is, t1rs);
		v4f moi0  = v4f_add(t0ia, t1ia);
		v4f moi2  = v4f_sub(t0ia, t1ia);

		v4f ptwr1 = v4f_ld(coefs + 32);
		v4f ptwi1 = v4f_ld(coefs + 36);
		v4f ptwr2 = v4f_ld(coefs + 40);
		v4f ptwi2 = v4f_ld(coefs + 44);
		v4f ptwr3 = v4f_ld(coefs + 48);
		v4f ptwi3 = v4f_ld(coefs + 52);
		v4f tor1a = v4f_mul(mor1, ptwr1);
		v4f toi1a = v4f_mul(mor1, ptwi1);
		v4f tor1b = v4f_mul(moi1, ptwi1);
		v4f toi1b = v4f_mul(moi1, ptwr1);
		v4f tor2a = v4f_mul(mor2, ptwr2);
		v4f toi2a = v4f_mul(mor2, ptwi2);
		v4f tor2b = v4f_mul(moi2, ptwi2);
		v4f toi2b = v4f_mul(moi2, ptwr2);
		v4f tor3a = v4f_mul(mor3, ptwr3);
		v4f toi3a = v4f_mul(mor3, ptwi3);
		v4f tor3b = v4f_mul(moi3, ptwi3);
		v4f toi3b = v4f_mul(moi3, ptwr3);
		v4f tor1  = v4f_sub(tor1a, tor1b);
		v4f toi1  = v4f_add(toi1a, toi1b);
		v4f tor2  = v4f_sub(tor2a, tor2b);
		v4f toi2  = v4f_add(toi2a, toi2b);
		v4f tor3  = v4f_sub(tor3a, tor3b);
		v4f toi3  = v4f_add(toi3a, toi3b);

		V4F_TRANSPOSE_INPLACE(mor0, tor1, tor2, tor3);
		V4F_TRANSPOSE_INPLACE(moi0, toi1, toi2, toi3);

		v4f_st(vec_output + 0,  mor0);
		v4f_st(vec_output + 4,  moi0);
		v4f_st(vec_output + 8,  tor1);
		v4f_st(vec_output + 12, toi1);
		v4f_st(vec_output + 16, tor2);
		v4f_st(vec_output + 20, toi2);
		v4f_st(vec_output + 24, tor3);
		v4f_st(vec_output + 28, toi3);
	}
}

static void modfreqoffsetreal_inverse_final(float *output, const float *vec_input, const float *coefs, unsigned fft_len)
{
	const unsigned fft_len_4 = fft_len / 4;
	unsigned i;
	assert((fft_len % 16) == 0);
	for (i = fft_len / 16
		;i
		;i--, vec_input += 32, coefs += 56, output += 4) {
		v4f r0   = v4f_ld(vec_input + 0);
		v4f i0   = v4f_ld(vec_input + 4);
		v4f r1   = v4f_ld(vec_input + 8);
		v4f i1   = v4f_ld(vec_input + 12);
		v4f r2   = v4f_ld(vec_input + 16);
		v4f i2   = v4f_ld(vec_input + 20);
		v4f r3   = v4f_ld(vec_input + 24);
		v4f i3   = v4f_ld(vec_input + 28);
		V4F_TRANSPOSE_INPLACE(r0, r1, r2, r3);
		V4F_TRANSPOSE_INPLACE(i0, i1, i2, i3);
		{
			v4f ptwr1 = v4f_ld(coefs + 32);
			v4f ptwi1 = v4f_ld(coefs + 36);
			v4f ptwr2 = v4f_ld(coefs + 40);
			v4f ptwi2 = v4f_ld(coefs + 44);
			v4f ptwr3 = v4f_ld(coefs + 48);
			v4f ptwi3 = v4f_ld(coefs + 52);
			v4f tor1a = v4f_mul(r1, ptwr1);
			v4f toi1a = v4f_mul(r1, ptwi1);
			v4f tor1b = v4f_mul(i1, ptwi1);
			v4f toi1b = v4f_mul(i1, ptwr1);
			v4f tor2a = v4f_mul(r2, ptwr2);
			v4f toi2a = v4f_mul(r2, ptwi2);
			v4f tor2b = v4f_mul(i2, ptwi2);
			v4f toi2b = v4f_mul(i2, ptwr2);
			v4f tor3a = v4f_mul(r3, ptwr3);
			v4f toi3a = v4f_mul(r3, ptwi3);
			v4f tor3b = v4f_mul(i3, ptwi3);
			v4f toi3b = v4f_mul(i3, ptwr3);
			v4f tor1  = v4f_sub(tor1a, tor1b);
			v4f toi1  = v4f_add(toi1a, toi1b);
			v4f tor2  = v4f_sub(tor2a, tor2b);
			v4f toi2  = v4f_add(toi2a, toi2b);
			v4f tor3  = v4f_sub(tor3a, tor3b);
			v4f toi3  = v4f_add(toi3a, toi3b);

			v4f t0ra  = v4f_add(r0,   tor2);
			v4f t0rs  = v4f_sub(r0,   tor2);
			v4f t1ra  = v4f_add(tor1, tor3);
			v4f t1rs  = v4f_sub(tor1, tor3);
			v4f t1is  = v4f_sub(toi1, toi3);
			v4f t1ia  = v4f_add(toi1, toi3);
			v4f t0is  = v4f_sub(i0,   toi2);
			v4f t0ia  = v4f_add(i0,   toi2);
			v4f mor0  = v4f_add(t0ra, t1ra);
			v4f mor2  = v4f_sub(t0ra, t1ra);
			v4f mor1  = v4f_add(t0rs, t1is);
			v4f mor3  = v4f_sub(t0rs, t1is);
			v4f moi1  = v4f_sub(t0is, t1rs);
			v4f moi3  = v4f_add(t0is, t1rs);
			v4f moi0  = v4f_add(t0ia, t1ia);
			v4f moi2  = v4f_sub(t0ia, t1ia);

			v4f twr1  = v4f_ld(coefs + 0);
			v4f twi1  = v4f_ld(coefs + 4);
			v4f twr2  = v4f_ld(coefs + 8);
			v4f twi2  = v4f_ld(coefs + 12);
			v4f or0a  = v4f_mul(twr1, mor0);
			v4f or0b  = v4f_mul(twi1, moi0);
			v4f oi0a  = v4f_mul(twi1, mor0);
			v4f oi0b  = v4f_mul(twr1, moi0);
			v4f or1a  = v4f_mul(twr2, mor1);
			v4f or1b  = v4f_mul(twi2, moi1);
			v4f oi1a  = v4f_mul(twi2, mor1);
			v4f oi1b  = v4f_mul(twr2, moi1);
			v4f or0   = v4f_sub(or0a, or0b);
			v4f oi0   = v4f_add(oi0a, oi0b);
			v4f or1   = v4f_sub(or1a, or1b);
			v4f oi1   = v4f_add(oi1a, oi1b);
			v4f twr3  = v4f_ld(coefs + 16);
			v4f twi3  = v4f_ld(coefs + 20);
			v4f twr4  = v4f_ld(coefs + 24);
			v4f twi4  = v4f_ld(coefs + 28);
			v4f or2a  = v4f_mul(twr3, mor2);
			v4f or2b  = v4f_mul(twi3, moi2);
			v4f oi2a  = v4f_mul(twi3, mor2);
			v4f oi2b  = v4f_mul(twr3, moi2);
			v4f or3a  = v4f_mul(twr4, mor3);
			v4f or3b  = v4f_mul(twi4, moi3);
			v4f oi3a  = v4f_mul(twi4, mor3);
			v4f oi3b  = v4f_mul(twr4, moi3);
			v4f or2   = v4f_sub(or2a, or2b);
			v4f oi2   = v4f_add(oi2a, oi2b);
			v4f or3   = v4f_sub(or3a, or3b);
			v4f oi3   = v4f_add(oi3a, oi3b);

			v4f_st(output + 0*fft_len_4, or0);
			v4f_st(output + 1*fft_len_4, or1);
			v4f_st(output + 2*fft_len_4, or2);
			v4f_st(output + 3*fft_len_4, or3);
			v4f_st(output + 4*fft_len_4, oi0);
			v4f_st(output + 5*fft_len_4, oi1);
			v4f_st(output + 6*fft_len_4, oi2);
			v4f_st(output + 7*fft_len_4, oi3);
		}
	}
}

static void modfreqoffsetreal_forward_reorder(float *out_buf, const float *in_buf, const float *twid, unsigned lfft)
{
	unsigned i;
	for (i = 0; i < lfft / 8; i++) {
		v4f tor1, toi1, tor2, toi2; 
		v4f re1 = v4f_ld(in_buf + i*8 + 0);
		v4f im1 = v4f_ld(in_buf + i*8 + 4);
		v4f re2 = v4f_ld(in_buf + lfft*2 - i*8 - 8);
		v4f im2 = v4f_ld(in_buf + lfft*2 - i*8 - 4);
		re2     = v4f_reverse(re2);
		im2     = v4f_reverse(v4f_neg(im2));
		V4F_INTERLEAVE(tor1, tor2, re1, re2);
		V4F_INTERLEAVE(toi1, toi2, im1, im2);
		V4F_ST2X2INT(out_buf + i*16, out_buf + i*16 + 8, tor1, toi1, tor2, toi2);
	}
}

static void modfreqoffsetreal_inverse_reorder(float *out_buf, const float *in_buf, const float *twid, unsigned lfft)
{
	unsigned i;
	for (i = 0; i < lfft / 8; i++) {
		v4f tor1, toi1, tor2, toi2, re1, im1, re2, im2;
		V4F_LD2X2DINT(tor1, toi1, tor2, toi2, in_buf + i*16, in_buf + i*16 + 8);
		V4F_DEINTERLEAVE(re1, re2, tor1, tor2);
		V4F_DEINTERLEAVE(im1, im2, toi1, toi2);
		re2 = v4f_reverse(re2);
		im2 = v4f_reverse(im2);
		im1 = v4f_neg(im1);
		v4f_st(out_buf + i*8 + 0, re1);
		v4f_st(out_buf + i*8 + 4, im1);
		v4f_st(out_buf + lfft*2 - i*8 - 8, re2);
		v4f_st(out_buf + lfft*2 - i*8 - 4, im2);
	}
}

static const float *modfreqoffsetreal_get_twid(struct aalloc *alloc, unsigned length)
{
	float *twid = aalloc_align_alloc(alloc, sizeof(float) * 56 * length / 16, 64);
	if (twid != NULL) {
		unsigned i;
		for (i = 0; i < length / 16; i++) {
			twid[56*i+0]  = cosf((4*i+0) * -0.5 * M_PI / length);
			twid[56*i+1]  = cosf((4*i+1) * -0.5 * M_PI / length);
			twid[56*i+2]  = cosf((4*i+2) * -0.5 * M_PI / length);
			twid[56*i+3]  = cosf((4*i+3) * -0.5 * M_PI / length);
			twid[56*i+4]  = sinf((4*i+0) * -0.5 * M_PI / length);
			twid[56*i+5]  = sinf((4*i+1) * -0.5 * M_PI / length);
			twid[56*i+6]  = sinf((4*i+2) * -0.5 * M_PI / length);
			twid[56*i+7]  = sinf((4*i+3) * -0.5 * M_PI / length);
			twid[56*i+8]  = cosf((4*i+0+1*length/4) * -0.5 * M_PI / length);
			twid[56*i+9]  = cosf((4*i+1+1*length/4) * -0.5 * M_PI / length);
			twid[56*i+10] = cosf((4*i+2+1*length/4) * -0.5 * M_PI / length);
			twid[56*i+11] = cosf((4*i+3+1*length/4) * -0.5 * M_PI / length);
			twid[56*i+12] = sinf((4*i+0+1*length/4) * -0.5 * M_PI / length);
			twid[56*i+13] = sinf((4*i+1+1*length/4) * -0.5 * M_PI / length);
			twid[56*i+14] = sinf((4*i+2+1*length/4) * -0.5 * M_PI / length);
			twid[56*i+15] = sinf((4*i+3+1*length/4) * -0.5 * M_PI / length);
			twid[56*i+16] = cosf((4*i+0+2*length/4) * -0.5 * M_PI / length);
			twid[56*i+17] = cosf((4*i+1+2*length/4) * -0.5 * M_PI / length);
			twid[56*i+18] = cosf((4*i+2+2*length/4) * -0.5 * M_PI / length);
			twid[56*i+19] = cosf((4*i+3+2*length/4) * -0.5 * M_PI / length);
			twid[56*i+20] = sinf((4*i+0+2*length/4) * -0.5 * M_PI / length);
			twid[56*i+21] = sinf((4*i+1+2*length/4) * -0.5 * M_PI / length);
			twid[56*i+22] = sinf((4*i+2+2*length/4) * -0.5 * M_PI / length);
			twid[56*i+23] = sinf((4*i+3+2*length/4) * -0.5 * M_PI / length);
			twid[56*i+24] = cosf((4*i+0+3*length/4) * -0.5 * M_PI / length);
			twid[56*i+25] = cosf((4*i+1+3*length/4) * -0.5 * M_PI / length);
			twid[56*i+26] = cosf((4*i+2+3*length/4) * -0.5 * M_PI / length);
			twid[56*i+27] = cosf((4*i+3+3*length/4) * -0.5 * M_PI / length);
			twid[56*i+28] = sinf((4*i+0+3*length/4) * -0.5 * M_PI / length);
			twid[56*i+29] = sinf((4*i+1+3*length/4) * -0.5 * M_PI / length);
			twid[56*i+30] = sinf((4*i+2+3*length/4) * -0.5 * M_PI / length);
			twid[56*i+31] = sinf((4*i+3+3*length/4) * -0.5 * M_PI / length);
			twid[56*i+32] = cosf(-2.0 * (4*i+0) * M_PI * 1.0 / length);
			twid[56*i+33] = cosf(-2.0 * (4*i+1) * M_PI * 1.0 / length);
			twid[56*i+34] = cosf(-2.0 * (4*i+2) * M_PI * 1.0 / length);
			twid[56*i+35] = cosf(-2.0 * (4*i+3) * M_PI * 1.0 / length);
			twid[56*i+36] = sinf(-2.0 * (4*i+0) * M_PI * 1.0 / length);
			twid[56*i+37] = sinf(-2.0 * (4*i+1) * M_PI * 1.0 / length);
			twid[56*i+38] = sinf(-2.0 * (4*i+2) * M_PI * 1.0 / length);
			twid[56*i+39] = sinf(-2.0 * (4*i+3) * M_PI * 1.0 / length);
			twid[56*i+40] = cosf(-2.0 * (4*i+0) * M_PI * 2.0 / length);
			twid[56*i+41] = cosf(-2.0 * (4*i+1) * M_PI * 2.0 / length);
			twid[56*i+42] = cosf(-2.0 * (4*i+2) * M_PI * 2.0 / length);
			twid[56*i+43] = cosf(-2.0 * (4*i+3) * M_PI * 2.0 / length);
			twid[56*i+44] = sinf(-2.0 * (4*i+0) * M_PI * 2.0 / length);
			twid[56*i+45] = sinf(-2.0 * (4*i+1) * M_PI * 2.0 / length);
			twid[56*i+46] = sinf(-2.0 * (4*i+2) * M_PI * 2.0 / length);
			twid[56*i+47] = sinf(-2.0 * (4*i+3) * M_PI * 2.0 / length);
			twid[56*i+48] = cosf(-2.0 * (4*i+0) * M_PI * 3.0 / length);
			twid[56*i+49] = cosf(-2.0 * (4*i+1) * M_PI * 3.0 / length);
			twid[56*i+50] = cosf(-2.0 * (4*i+2) * M_PI * 3.0 / length);
			twid[56*i+51] = cosf(-2.0 * (4*i+3) * M_PI * 3.0 / length);
			twid[56*i+52] = sinf(-2.0 * (4*i+0) * M_PI * 3.0 / length);
			twid[56*i+53] = sinf(-2.0 * (4*i+1) * M_PI * 3.0 / length);
			twid[56*i+54] = sinf(-2.0 * (4*i+2) * M_PI * 3.0 / length);
			twid[56*i+55] = sinf(-2.0 * (4*i+3) * M_PI * 3.0 / length);
		}
	}
	return twid;
}

static const struct fftset_modulation FFTSET_MODULATION_FREQ_OFFSET_REAL_DEF =
{   4
,   NULL
,   modfreqoffsetreal_get_twid
,   modfreqoffsetreal_forward_first
,   modfreqoffsetreal_forward_reorder
,   modfreqoffsetreal_inverse_reorder
,   modfreqoffsetreal_inverse_final
};

const struct fftset_modulation *FFTSET_MODULATION_FREQ_OFFSET_REAL = &FFTSET_MODULATION_FREQ_OFFSET_REAL_DEF;
