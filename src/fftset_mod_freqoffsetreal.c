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

#ifdef V4F_EXISTS
#define VEC_V4F_WIDTH (4)

#if V8F_EXISTS
static void modfreqoffsetreal_forward_first_v8f(float *vo, const float *input, const float *coefs, unsigned fft_len)
{
	const unsigned fft_len_4 = fft_len / 4;
	unsigned i;
	assert((fft_len % 32) == 0);

	float *vec_output = vo;

	for (i = fft_len / 32
		;i
		;i--, coefs += 56, vec_output += 64, input += 4) {
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
		v4f_st(vec_output + 16, tor1);
		v4f_st(vec_output + 20, toi1);
		v4f_st(vec_output + 32, tor2);
		v4f_st(vec_output + 36, toi2);
		v4f_st(vec_output + 48, tor3);
		v4f_st(vec_output + 52, toi3);
	}

	vec_output = vo;

	for (i = fft_len / 32
		;i
		;i--, coefs += 56, vec_output += 64, input += 4) {
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

		v4f_st(vec_output + 8,  mor0);
		v4f_st(vec_output + 12, moi0);
		v4f_st(vec_output + 24, tor1);
		v4f_st(vec_output + 28, toi1);
		v4f_st(vec_output + 40, tor2);
		v4f_st(vec_output + 44, toi2);
		v4f_st(vec_output + 56, tor3);
		v4f_st(vec_output + 60, toi3);
	}

	vec_output = vo;

	for (i = 0
		;i < fft_len / 8
		;i++, vec_output += 16, coefs += 2) {
		v4f r0, i0, r1, i1;
		v4f twr, twi;
		v4f r2, i2, r3, i3;
		v4f or0, or1, oi0, oi1;
		v4f xr1, xi1;
		V4F_LD2(r0, i0, vec_output);
		V4F_LD2(r1, i1, vec_output + 8);
		twr = v4f_broadcast(coefs[0]);
		twi = v4f_broadcast(coefs[1]);
		xr1 = v4f_sub(r0, r1);
		xi1 = v4f_sub(i0, i1);
		or0 = v4f_add(r0, r1);
		oi0 = v4f_add(i0, i1);
		r2  = v4f_mul(xr1, twr);
		i2  = v4f_mul(xi1, twr);
		r3  = v4f_mul(xi1, twi);
		i3  = v4f_mul(xr1, twi);
		or1 = v4f_sub(r2, r3);
		oi1 = v4f_add(i2, i3);
		V4F_ST2(vec_output,     or0, or1);
		V4F_ST2(vec_output + 8, oi0, oi1);
	}
}

static void modfreqoffsetreal_inverse_final_v8f(float *output, const float *vi, const float *coefs, unsigned fft_len)
{
	const unsigned fft_len_8 = fft_len / 8;
	const unsigned fft_len_4 = fft_len / 4;
	unsigned i;
	const float *vec_input = vi;
	float *vec_output = output;
	const float *tp = coefs + 56 * fft_len / 16;

	assert((fft_len % 32) == 0);

	for (i = 0
		;i < fft_len / 32
		;i++, vec_input += 64, vec_output += 4, tp += 8) {
		v4f r0, i0, r1, i1;
		v4f twr0, twi0;
		v4f twr1, twi1;
		v4f r2, i2, r3, i3;
		v4f or0, or1, oi0, oi1;
		v4f or2, or3, oi2, oi3;

		V4F_LD2(or0, or1, vec_input);
		V4F_LD2(oi0, oi1, vec_input + 8);
		V4F_LD2(or2, or3, vec_input + 16);
		V4F_LD2(oi2, oi3, vec_input + 24);
		twr0 = v4f_broadcast(tp[0]);
		twi0 = v4f_broadcast(tp[1]);
		twr1 = v4f_broadcast(tp[2]);
		twi1 = v4f_broadcast(tp[3]);
		r0  = v4f_mul(or1, twr0);
		i0  = v4f_mul(oi1, twr0);
		r1  = v4f_mul(oi1, twi0);
		i1  = v4f_mul(or1, twi0);
		r2  = v4f_mul(or3, twr1);
		i2  = v4f_mul(oi3, twr1);
		r3  = v4f_mul(oi3, twi1);
		i3  = v4f_mul(or3, twi1);
		or1 = v4f_sub(r0, r1);
		oi1 = v4f_add(i0, i1);
		or3 = v4f_sub(r2, r3);
		oi3 = v4f_add(i2, i3);
		r0  = v4f_add(or0, or1);
		i0  = v4f_add(oi0, oi1);
		r1  = v4f_sub(or0, or1);
		i1  = v4f_sub(oi0, oi1);
		r2  = v4f_add(or2, or3);
		i2  = v4f_add(oi2, oi3);
		r3  = v4f_sub(or2, or3);
		i3  = v4f_sub(oi2, oi3);
		v4f_st(vec_output + 0*fft_len_8, r0);
		v4f_st(vec_output + 1*fft_len_8, r1);
		v4f_st(vec_output + 2*fft_len_8, i0);
		v4f_st(vec_output + 3*fft_len_8, i1);
		v4f_st(vec_output + 4*fft_len_8, r2);
		v4f_st(vec_output + 5*fft_len_8, r3);
		v4f_st(vec_output + 6*fft_len_8, i2);
		v4f_st(vec_output + 7*fft_len_8, i3);

		V4F_LD2(or0, or1, vec_input + 32);
		V4F_LD2(oi0, oi1, vec_input + 40);
		V4F_LD2(or2, or3, vec_input + 48);
		V4F_LD2(oi2, oi3, vec_input + 56);
		twr0 = v4f_broadcast(tp[4]);
		twi0 = v4f_broadcast(tp[5]);
		twr1 = v4f_broadcast(tp[6]);
		twi1 = v4f_broadcast(tp[7]);
		r0  = v4f_mul(or1, twr0);
		i0  = v4f_mul(oi1, twr0);
		r1  = v4f_mul(oi1, twi0);
		i1  = v4f_mul(or1, twi0);
		r2  = v4f_mul(or3, twr1);
		i2  = v4f_mul(oi3, twr1);
		r3  = v4f_mul(oi3, twi1);
		i3  = v4f_mul(or3, twi1);
		or1 = v4f_sub(r0, r1);
		oi1 = v4f_add(i0, i1);
		or3 = v4f_sub(r2, r3);
		oi3 = v4f_add(i2, i3);
		r0  = v4f_add(or0, or1);
		i0  = v4f_add(oi0, oi1);
		r1  = v4f_sub(or0, or1);
		i1  = v4f_sub(oi0, oi1);
		r2  = v4f_add(or2, or3);
		i2  = v4f_add(oi2, oi3);
		r3  = v4f_sub(or2, or3);
		i3  = v4f_sub(oi2, oi3);
		v4f_st(vec_output + 8*fft_len_8, r0);
		v4f_st(vec_output + 9*fft_len_8, r1);
		v4f_st(vec_output + 10*fft_len_8, i0);
		v4f_st(vec_output + 11*fft_len_8, i1);
		v4f_st(vec_output + 12*fft_len_8, r2);
		v4f_st(vec_output + 13*fft_len_8, r3);
		v4f_st(vec_output + 14*fft_len_8, i2);
		v4f_st(vec_output + 15*fft_len_8, i3);
	}

	for (i = fft_len / 16
		;i
		;i--, coefs += 56, output += 4) {
		v4f r0   = v4f_ld(output + 0*fft_len_4);
		v4f i0   = v4f_ld(output + 1*fft_len_4);
		v4f r1   = v4f_ld(output + 2*fft_len_4);
		v4f i1   = v4f_ld(output + 3*fft_len_4);
		v4f r2   = v4f_ld(output + 4*fft_len_4);
		v4f i2   = v4f_ld(output + 5*fft_len_4);
		v4f r3   = v4f_ld(output + 6*fft_len_4);
		v4f i3   = v4f_ld(output + 7*fft_len_4);
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

static
void
modfreqoffsetreal_get_kernel_v8f
	(const struct fftset_fft    *first_pass
	,float                      *output_buf
	,const float                *input_buf
	)
{
	modfreqoffsetreal_forward_first_v8f(output_buf, input_buf, first_pass->main_twiddle, first_pass->lfft);
	fftset_vec_kern(first_pass->next_compat, 1, output_buf);
}

static
void
modfreqoffsetreal_conv_v8f
	(const struct fftset_fft *first_pass
	,float                      *output_buf
	,const float                *input_buf
	,const float                *kernel_buf
	,float                      *work_buf
	)
{
	modfreqoffsetreal_forward_first_v8f(work_buf, input_buf, first_pass->main_twiddle, first_pass->lfft);
	fftset_vec_conv(first_pass->next_compat, 1, work_buf, kernel_buf);
	modfreqoffsetreal_inverse_final_v8f(output_buf, work_buf, first_pass->main_twiddle, first_pass->lfft);
}

static
void
modfreqoffsetreal_forward_v8f
	(const struct fftset_fft *first_pass
	,float                      *output_buf
	,const float                *input_buf
	,float                      *work_buf
	)
{
	const unsigned lfft = first_pass->lfft;
	unsigned i;

	modfreqoffsetreal_forward_first_v8f(work_buf, input_buf, first_pass->main_twiddle, lfft);

	input_buf = fftset_vec_stockham(first_pass->next_compat, 1, work_buf, output_buf);

	if (input_buf != work_buf) {
		assert(input_buf == output_buf);
		memcpy(work_buf, output_buf, sizeof(float) * lfft * 2);
	}

	for (i = 0; i < lfft / 16; i++) {
		v8f w, x, y, z;
		v8f a, b, c, d;
		V8F_LD2(w, x, work_buf + i*16);
		V8F_LD2(y, z, work_buf + lfft*2 - 16 - i*16);
		y = v8f_reverse(y);
		z = v8f_reverse(z);
		z = v8f_neg(z);
		V8F_INTERLEAVE(a, b, w, y);
		V8F_INTERLEAVE(c, d, x, z);
		V8F_ST2X2INT(output_buf + i*32, output_buf + i*32 + 16, a, c, b, d);
	}
}

static
void
modfreqoffsetreal_inverse_v8f
	(const struct fftset_fft    *first_pass
	,float                      *output_buf
	,const float                *input_buf
	,float                      *work_buf
	)
{
	const unsigned lfft = first_pass->lfft;
	unsigned i;

	for (i = 0; i < lfft / 16; i++) {
		v8f w, x, y, z;
		v8f a, b, c, d;
		V8F_LD2X2DINT(a, c, b, d, input_buf + i*32, input_buf + i*32 + 16);
		V8F_DEINTERLEAVE(w, y, a, b);
		V8F_DEINTERLEAVE(x, z, c, d);
		x = v8f_neg(x);
		y = v8f_reverse(y);
		z = v8f_reverse(z);
		V8F_ST2(work_buf + i*16,               w, x);
		V8F_ST2(work_buf + lfft*2 - 16 - i*16, y, z);
	}

	input_buf = fftset_vec_stockham(first_pass->next_compat, 1, work_buf, output_buf);

	if (input_buf == work_buf) {
		modfreqoffsetreal_inverse_final_v8f(output_buf, work_buf, first_pass->main_twiddle, lfft);
	} else {
		assert(input_buf == output_buf);
		memcpy(work_buf, output_buf, sizeof(float) * lfft * 2);
		modfreqoffsetreal_inverse_final_v8f(output_buf, work_buf, first_pass->main_twiddle, lfft);
	}
}

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

static
void
modfreqoffsetreal_get_kernel_v4f
	(const struct fftset_fft    *first_pass
	,float                      *output_buf
	,const float                *input_buf
	)
{
	modfreqoffsetreal_forward_first(output_buf, input_buf, first_pass->main_twiddle, first_pass->lfft);
	fftset_vec_kern(first_pass->next_compat, 1, output_buf);
}

static
void
modfreqoffsetreal_conv_v4f
	(const struct fftset_fft *first_pass
	,float                      *output_buf
	,const float                *input_buf
	,const float                *kernel_buf
	,float                      *work_buf
	)
{
	modfreqoffsetreal_forward_first(work_buf, input_buf, first_pass->main_twiddle, first_pass->lfft);
	fftset_vec_conv(first_pass->next_compat, 1, work_buf, kernel_buf);
	modfreqoffsetreal_inverse_final(output_buf, work_buf, first_pass->main_twiddle, first_pass->lfft);
}

static
void
modfreqoffsetreal_forward_v4f
	(const struct fftset_fft *first_pass
	,float                      *output_buf
	,const float                *input_buf
	,float                      *work_buf
	)
{
	const unsigned lfft = first_pass->lfft;
	unsigned i;

	modfreqoffsetreal_forward_first(work_buf, input_buf, first_pass->main_twiddle, lfft);

	input_buf = fftset_vec_stockham(first_pass->next_compat, 1, work_buf, output_buf);

	if (input_buf != work_buf) {
		assert(input_buf == output_buf);
		memcpy(work_buf, output_buf, sizeof(float) * lfft * 2);
	}

	for (i = 0; i < lfft / 8; i++) {
		v4f tor1, toi1, tor2, toi2; 
		v4f re1 = v4f_ld(work_buf + i*8 + 0);
		v4f im1 = v4f_ld(work_buf + i*8 + 4);
		v4f re2 = v4f_ld(work_buf + lfft*2 - i*8 - 8);
		v4f im2 = v4f_ld(work_buf + lfft*2 - i*8 - 4);
		re2     = v4f_reverse(re2);
		im2     = v4f_reverse(v4f_neg(im2));
		V4F_INTERLEAVE(tor1, tor2, re1, re2);
		V4F_INTERLEAVE(toi1, toi2, im1, im2);
		V4F_ST2X2INT(output_buf + i*16, output_buf + i*16 + 8, tor1, toi1, tor2, toi2);
	}
}

static
void
modfreqoffsetreal_inverse_v4f
	(const struct fftset_fft    *first_pass
	,float                      *output_buf
	,const float                *input_buf
	,float                      *work_buf
	)
{
	const unsigned lfft = first_pass->lfft;
	unsigned i;

	for (i = 0; i < lfft / 8; i++) {
		v4f tor1, toi1, tor2, toi2, re1, im1, re2, im2;
		V4F_LD2X2DINT(tor1, toi1, tor2, toi2, input_buf + i*16, input_buf + i*16 + 8);
		V4F_DEINTERLEAVE(re1, re2, tor1, tor2);
		V4F_DEINTERLEAVE(im1, im2, toi1, toi2);
		re2 = v4f_reverse(re2);
		im2 = v4f_reverse(im2);
		im1 = v4f_neg(im1);
		v4f_st(work_buf + i*8 + 0, re1);
		v4f_st(work_buf + i*8 + 4, im1);
		v4f_st(work_buf + lfft*2 - i*8 - 8, re2);
		v4f_st(work_buf + lfft*2 - i*8 - 4, im2);
	}

	input_buf = fftset_vec_stockham(first_pass->next_compat, 1, work_buf, output_buf);

	if (input_buf == work_buf) {
		modfreqoffsetreal_inverse_final(output_buf, work_buf, first_pass->main_twiddle, lfft);
	} else {
		assert(input_buf == output_buf);
		memcpy(work_buf, output_buf, sizeof(float) * lfft * 2);
		modfreqoffsetreal_inverse_final(output_buf, work_buf, first_pass->main_twiddle, lfft);
	}
}
#endif

static
void
modfreqoffsetreal_get_kernel_v1f
	(const struct fftset_fft    *first_pass
	,float                      *output_buf
	,const float                *input_buf
	)
{
	unsigned i;
	unsigned lfft = first_pass->lfft;
	for (i = 0; i < lfft; i++) {
		float re  =  input_buf[i];
		float im  = -input_buf[lfft+i];
		float twr = cosf(-2.0f * (float)M_PI * i / (lfft * 4.0f));
		float twi = sinf(-2.0f * (float)M_PI * i / (lfft * 4.0f));
		output_buf[2*i+0] = re * twr - im * twi;
		output_buf[2*i+1] = re * twi + im * twr;
	}
	fftset_vec_kern(first_pass->next_compat, 1, output_buf);
}

static
void
modfreqoffsetreal_conv_v1f
	(const struct fftset_fft *first_pass
	,float                   *output_buf
	,const float             *input_buf
	,const float             *kernel_buf
	,float                   *work_buf
	)
{
	unsigned i;
	unsigned lfft = first_pass->lfft;
	for (i = 0; i < lfft; i++) {
		float re        = input_buf[i];
		float im        = input_buf[lfft+i];
		float twr       = cosf(-2.0f * (float)M_PI * i / (lfft * 4.0f));
		float twi       = sinf(-2.0f * (float)M_PI * i / (lfft * 4.0f));
		work_buf[2*i+0] = re * twr + im * twi;
		work_buf[2*i+1] = re * twi - im * twr;
	}
	fftset_vec_conv(first_pass->next_compat, 1, work_buf, kernel_buf);
	for (i = 0; i < lfft; i++) {
		float re           = work_buf[2*i+0];
		float im           = work_buf[2*i+1];
		float twr          = cosf(-2.0f * (float)M_PI * i / (lfft * 4.0f));
		float twi          = sinf(-2.0f * (float)M_PI * i / (lfft * 4.0f));
		output_buf[i]      = re * twr - im * twi;
		output_buf[lfft+i] = re * twi + im * twr;
	}
}

static
void
modfreqoffsetreal_forward_v1f
	(const struct fftset_fft *first_pass
	,float                   *output_buf
	,const float             *input_buf
	,float                   *work_buf
	)
{
	unsigned i;
	unsigned lfft = first_pass->lfft;
	for (i = 0; i < lfft; i++) {
		float re        = input_buf[i];
		float im        = input_buf[lfft+i];
		float twr       = cosf(-2.0f * (float)M_PI * i / (lfft * 4.0f));
		float twi       = sinf(-2.0f * (float)M_PI * i / (lfft * 4.0f));
		work_buf[2*i+0] = re * twr + im * twi;
		work_buf[2*i+1] = re * twi - im * twr;
	}
	input_buf = fftset_vec_stockham(first_pass->next_compat, 1, work_buf, output_buf);
	if (input_buf == output_buf)
		memcpy(work_buf, output_buf, sizeof(float) * lfft * 2);
	for (i = 0; i < lfft / 2; i++) {
		float re0         = work_buf[2*i+0];
		float im0         = work_buf[2*i+1];
		float re1         = work_buf[2*lfft-2-2*i];
		float im1         = work_buf[2*lfft-1-2*i];
		output_buf[4*i+0] = re0;
		output_buf[4*i+1] = im0;
		output_buf[4*i+2] = re1;
		output_buf[4*i+3] = -im1;
	}
	if (lfft & 1) {
		output_buf[4*i+0] = work_buf[2*i+0];
		output_buf[4*i+1] = work_buf[2*i+1];
	}
}

static
void
modfreqoffsetreal_inverse_v1f
	(const struct fftset_fft    *first_pass
	,float                      *output_buf
	,const float                *input_buf
	,float                      *work_buf
	)
{
	unsigned i;
	unsigned lfft = first_pass->lfft;
	for (i = 0; i < lfft / 2; i++) {
		float re0 = input_buf[4*i+0];
		float im0 = input_buf[4*i+1];
		float re1 = input_buf[4*i+2];
		float im1 = input_buf[4*i+3];
		work_buf[2*i+0]        = re0;
		work_buf[2*i+1]        = -im0;
		work_buf[2*lfft-2-2*i] = re1;
		work_buf[2*lfft-1-2*i] = im1;
	}
	if (lfft & 1) {
		work_buf[2*i+0] = output_buf[4*i+0];
		work_buf[2*i+1] = -output_buf[4*i+1];
	}
	input_buf = fftset_vec_stockham(first_pass->next_compat, 1, work_buf, output_buf);
	if (input_buf == output_buf)
		memcpy(work_buf, output_buf, sizeof(float) * lfft * 2);
	for (i = 0; i < lfft; i++) {
		float re           = work_buf[2*i+0];
		float im           = work_buf[2*i+1];
		float twr          = cosf(-2.0f * (float)M_PI * i / (lfft * 4.0f));
		float twi          = sinf(-2.0f * (float)M_PI * i / (lfft * 4.0f));
		output_buf[i]      = re * twr - im * twi;
		output_buf[lfft+i] = re * twi + im * twr;
	}
}

static int modfreqoffsetreal_init(struct fftset_fft *fft, struct fftset_vec **veclist, struct cop_salloc_iface *alloc, unsigned complex_len)
{
#if V4F_EXISTS
#if V8F_EXISTS
	if (complex_len >= 32 && complex_len % 32 == 0) {
		static const float off = (float)(-M_PI * 0.125);
		unsigned i;
		float *twid;
		float scale;

		/* Create inner passes. */
		fft->next_compat = fastconv_get_inner_pass(veclist, alloc, complex_len / 8, 8);
		if (fft->next_compat == NULL)
			return -1;

		/* Create memory for twiddle coefficients. */
#if V8F_EXISTS
		twid = cop_salloc(alloc, sizeof(float) * (56 * complex_len / 16 + 8 * complex_len / 32), 64);
#else
		twid = cop_salloc(alloc, sizeof(float) * (56 * complex_len / 16), 64);
#endif
		if (twid == NULL)
			return -1;

		scale = off / (float)(complex_len / 4);

		for (i = 0; i < complex_len / 4; i++) {
			float  fi            = i*scale;
			float *tp            = twid + ((i % VEC_V4F_WIDTH) + (i / VEC_V4F_WIDTH)*14*VEC_V4F_WIDTH);
			tp[0*VEC_V4F_WIDTH]  = cosf(fi+(0.0f*off));
			tp[1*VEC_V4F_WIDTH]  = sinf(fi+(0.0f*off));
			tp[2*VEC_V4F_WIDTH]  = cosf(fi+(1.0f*off));
			tp[3*VEC_V4F_WIDTH]  = sinf(fi+(1.0f*off));
			tp[4*VEC_V4F_WIDTH]  = cosf(fi+(2.0f*off));
			tp[5*VEC_V4F_WIDTH]  = sinf(fi+(2.0f*off));
			tp[6*VEC_V4F_WIDTH]  = cosf(fi+(3.0f*off));
			tp[7*VEC_V4F_WIDTH]  = sinf(fi+(3.0f*off));
			tp[8*VEC_V4F_WIDTH]  = cosf(fi * 4.0f * 1.0f);
			tp[9*VEC_V4F_WIDTH]  = sinf(fi * 4.0f * 1.0f);
			tp[10*VEC_V4F_WIDTH] = cosf(fi * 4.0f * 2.0f);
			tp[11*VEC_V4F_WIDTH] = sinf(fi * 4.0f * 2.0f);
			tp[12*VEC_V4F_WIDTH] = cosf(fi * 4.0f * 3.0f);
			tp[13*VEC_V4F_WIDTH] = sinf(fi * 4.0f * 3.0f);
		}

#if V8F_EXISTS
		for (i = 0; i < 4 * complex_len / 32; i++) {
			twid[56 * complex_len / 16 + 2*i + 0] = cosf(-2.0*i*M_PI/(complex_len/4));
			twid[56 * complex_len / 16 + 2*i + 1] = sinf(-2.0*i*M_PI/(complex_len/4));
		}
#endif


		fft->main_twiddle = twid;
		fft->get_kern     = modfreqoffsetreal_get_kernel_v8f;
		fft->fwd          = modfreqoffsetreal_forward_v8f;
		fft->inv          = modfreqoffsetreal_inverse_v8f;
		fft->conv         = modfreqoffsetreal_conv_v8f;
	}
	else
#endif
	if (complex_len >= 16 && complex_len % 16 == 0) {
		static const float off = (float)(-M_PI * 0.125);
		unsigned i;
		float *twid;
		float scale;

		/* Create inner passes. */
		fft->next_compat = fastconv_get_inner_pass(veclist, alloc, complex_len / 4, 4);
		if (fft->next_compat == NULL)
			return -1;

		/* Create memory for twiddle coefficients. */
		twid = cop_salloc(alloc, sizeof(float) * 56 * complex_len / 16, 64);
		if (twid == NULL)
			return -1;

		scale = off / (float)(complex_len / 4);

		for (i = 0; i < complex_len / 4; i++) {
			float  fi            = i*scale;
			float *tp            = twid + ((i % VEC_V4F_WIDTH) + (i / VEC_V4F_WIDTH)*14*VEC_V4F_WIDTH);
			tp[0*VEC_V4F_WIDTH]  = cosf(fi+(0.0f*off));
			tp[1*VEC_V4F_WIDTH]  = sinf(fi+(0.0f*off));
			tp[2*VEC_V4F_WIDTH]  = cosf(fi+(1.0f*off));
			tp[3*VEC_V4F_WIDTH]  = sinf(fi+(1.0f*off));
			tp[4*VEC_V4F_WIDTH]  = cosf(fi+(2.0f*off));
			tp[5*VEC_V4F_WIDTH]  = sinf(fi+(2.0f*off));
			tp[6*VEC_V4F_WIDTH]  = cosf(fi+(3.0f*off));
			tp[7*VEC_V4F_WIDTH]  = sinf(fi+(3.0f*off));
			tp[8*VEC_V4F_WIDTH]  = cosf(fi * 4.0f * 1.0f);
			tp[9*VEC_V4F_WIDTH]  = sinf(fi * 4.0f * 1.0f);
			tp[10*VEC_V4F_WIDTH] = cosf(fi * 4.0f * 2.0f);
			tp[11*VEC_V4F_WIDTH] = sinf(fi * 4.0f * 2.0f);
			tp[12*VEC_V4F_WIDTH] = cosf(fi * 4.0f * 3.0f);
			tp[13*VEC_V4F_WIDTH] = sinf(fi * 4.0f * 3.0f);
		}

		fft->main_twiddle = twid;
		fft->get_kern     = modfreqoffsetreal_get_kernel_v4f;
		fft->fwd          = modfreqoffsetreal_forward_v4f;
		fft->inv          = modfreqoffsetreal_inverse_v4f;
		fft->conv         = modfreqoffsetreal_conv_v4f;
	}
	else
#endif
	{
		fft->next_compat = fastconv_get_inner_pass(veclist, alloc, complex_len, 1);
		if (fft->next_compat == NULL)
			return -1;

		fft->main_twiddle = NULL;
		fft->get_kern     = modfreqoffsetreal_get_kernel_v1f;
		fft->fwd          = modfreqoffsetreal_forward_v1f;
		fft->inv          = modfreqoffsetreal_inverse_v1f;
		fft->conv         = modfreqoffsetreal_conv_v1f;
	}

	return 0;
}

static const struct fftset_modulation FFTSET_MODULATION_FREQ_OFFSET_REAL_DEF =
{   modfreqoffsetreal_init
};

const struct fftset_modulation *FFTSET_MODULATION_FREQ_OFFSET_REAL = &FFTSET_MODULATION_FREQ_OFFSET_REAL_DEF;

