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

#include "fftset_vec.h"
#include "cop/cop_vec.h"
#include <assert.h>
#include <math.h>

/* TODO... do something else */
#define FFT_VEC_LEN (4)

#if !defined(V4F_EXISTS)
/* What is the deal with this?
 * This will likely to change at some point when the cop vectors are extended
 * to support longer (or shorter) vector "vlf" type. All the external
 * modulators will need to change at this point also. */
#error "this module requires a v4f type at this point in time"
#endif

static const float C_C3 = 0.5f;               /* PI / 3 */
static const float C_S3 = 0.86602540378444f;  /* PI / 3 */
static const float C_C4 = 0.707106781186548f; /* PI / 4 */
static const float C_C8 = 0.923879532511288f; /* PI / 8 */
static const float C_S8 = 0.382683432365086f; /* PI / 8 */


static COP_ATTR_ALWAYSINLINE void fftset_vec_dif_fft2_offset_io(const float *in, float *out, const float *twid, unsigned in_stride, unsigned out_stride)
{
	v4f nre   = v4f_ld(in + 0);
	v4f nim   = v4f_ld(in + 4);
	v4f fre   = v4f_ld(in + in_stride + 0);
	v4f fim   = v4f_ld(in + in_stride + 4);
	v4f tre   = v4f_broadcast(twid[0]);
	v4f tim   = v4f_broadcast(twid[1]);
	v4f onre  = v4f_add(nre, fre);
	v4f onim  = v4f_add(nim, fim);
	v4f ptre  = v4f_sub(nre, fre);
	v4f ptim  = v4f_sub(nim, fim);
	v4f ofrea = v4f_mul(ptre, tre);
	v4f ofreb = v4f_mul(ptim, tim);
	v4f ofima = v4f_mul(ptre, tim);
	v4f ofimb = v4f_mul(ptim, tre);
	v4f ofre  = v4f_sub(ofrea, ofreb);
	v4f ofim  = v4f_add(ofima, ofimb);
	v4f_st(out + 0, onre);
	v4f_st(out + 4, onim);
	v4f_st(out + out_stride + 0, ofre);
	v4f_st(out + out_stride + 4, ofim);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dif_fft2_offset_o(const float *in, float *out, unsigned out_stride)
{
	v4f nre, nim, fre, fim;
	v4f onre, onim, ofre, ofim;
	V4F_LD2(nre, nim, in + 0);
	V4F_LD2(fre, fim, in + 8);
	onre = v4f_add(nre, fre);
	onim = v4f_add(nim, fim);
	ofre = v4f_sub(nre, fre);
	ofim = v4f_sub(nim, fim);
	V4F_ST2(out + 0, onre, onim);
	V4F_ST2(out + out_stride, ofre, ofim);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dit_fft2_offset_io(const float *in, float *out, const float *twid, unsigned in_stride, unsigned out_stride)
{
	v4f nre  = v4f_ld(in + 0);
	v4f nim  = v4f_ld(in + 4);
	v4f ptre = v4f_ld(in + in_stride + 0);
	v4f ptim = v4f_ld(in + in_stride + 4);
	v4f tre  = v4f_broadcast(twid[0]);
	v4f tim  = v4f_broadcast(twid[1]);
	v4f frea = v4f_mul(ptre, tre);
	v4f freb = v4f_mul(ptim, tim);
	v4f fima = v4f_mul(ptre, tim);
	v4f fimb = v4f_mul(ptim, tre);
	v4f fre  = v4f_sub(frea, freb);
	v4f fim  = v4f_add(fima, fimb);
	v4f onre = v4f_add(nre, fre);
	v4f onim = v4f_add(nim, fim);
	v4f ofre = v4f_sub(nre, fre);
	v4f ofim = v4f_sub(nim, fim);
	v4f_st(out + 0, onre);
	v4f_st(out + 4, onim);
	v4f_st(out + out_stride + 0, ofre);
	v4f_st(out + out_stride + 4, ofim);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dif_fft3_offset_io(const float *in, float *out, const float *twid, unsigned in_stride, unsigned out_stride)
{
	v4f r0, i0, r1, i1, r2, i2;
	v4f or0, oi0, or1, oi1, or2, oi2;
	v4f ar1, ai1, ar2, ai2;
	v4f cr1, ci1, cr2, ci2;
	v4f dr1, di1, dr2, di2;
	v4f er1, ei1, er2, ei2;
	v4f tr1, ti1, tr2, ti2;
	v4f tr3, ti3, tr4, ti4;
	v4f tr5, ti5;
	V4F_LD2(r0, i0, in + 0*in_stride);
	V4F_LD2(r1, i1, in + 1*in_stride);
	V4F_LD2(r2, i2, in + 2*in_stride);
	tr1 = v4f_add(r2, r1);
	ti1 = v4f_add(i2, i1);
	tr2 = v4f_sub(i2, i1);
	ti2 = v4f_sub(r2, r1);
	tr5 = v4f_mul(tr1, C_C3);
	ti5 = v4f_mul(ti1, C_C3);
	tr3 = v4f_mul(tr2, C_S3);
	ti3 = v4f_mul(ti2, C_S3);
	tr4 = v4f_sub(r0, tr5);
	ti4 = v4f_sub(i0, ti5);
	or0 = v4f_add(r0, tr1);
	oi0 = v4f_add(i0, ti1);
	ar1 = v4f_sub(tr4, tr3);
	ai1 = v4f_add(ti4, ti3);
	ar2 = v4f_add(tr4, tr3);
	ai2 = v4f_sub(ti4, ti3);
	cr1 = v4f_broadcast(twid[0]);
	ci1 = v4f_broadcast(twid[1]);
	cr2 = v4f_broadcast(twid[2]);
	ci2 = v4f_broadcast(twid[3]);
	dr1 = v4f_mul(ar1, cr1);
	er1 = v4f_mul(ai1, ci1);
	di1 = v4f_mul(ar1, ci1);
	ei1 = v4f_mul(ai1, cr1);
	dr2 = v4f_mul(ar2, cr2);
	er2 = v4f_mul(ai2, ci2);
	di2 = v4f_mul(ar2, ci2);
	ei2 = v4f_mul(ai2, cr2);
	or1 = v4f_sub(dr1, er1);
	oi1 = v4f_add(di1, ei1);
	or2 = v4f_sub(dr2, er2);
	oi2 = v4f_add(di2, ei2);
	V4F_ST2(out + 0*out_stride, or0, oi0);
	V4F_ST2(out + 1*out_stride, or1, oi1);
	V4F_ST2(out + 2*out_stride, or2, oi2);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dif_fft3_offset_o(const float *in, float *out, unsigned out_stride)
{
	v4f r0, i0, r1, i1, r2, i2;
	v4f or0, oi0, or1, oi1, or2, oi2;
	v4f tr1, ti1, tr2, ti2;
	v4f tr3, ti3, tr4, ti4;
	v4f tr5, ti5;
	V4F_LD2(r0, i0, in + 0);
	V4F_LD2(r1, i1, in + 8);
	V4F_LD2(r2, i2, in + 16);
	tr1 = v4f_add(r2, r1);
	ti1 = v4f_add(i2, i1);
	tr2 = v4f_sub(i2, i1);
	ti2 = v4f_sub(r2, r1);
	tr5 = v4f_mul(tr1, C_C3);
	ti5 = v4f_mul(ti1, C_C3);
	tr3 = v4f_mul(tr2, C_S3);
	ti3 = v4f_mul(ti2, C_S3);
	tr4 = v4f_sub(r0, tr5);
	ti4 = v4f_sub(i0, ti5);
	or0 = v4f_add(r0, tr1);
	oi0 = v4f_add(i0, ti1);
	or1 = v4f_sub(tr4, tr3);
	oi1 = v4f_add(ti4, ti3);
	or2 = v4f_add(tr4, tr3);
	oi2 = v4f_sub(ti4, ti3);
	V4F_ST2(out + 0*out_stride, or0, oi0);
	V4F_ST2(out + 1*out_stride, or1, oi1);
	V4F_ST2(out + 2*out_stride, or2, oi2);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dit_fft3_offset_io(const float *in, float *out, const float *twid, unsigned in_stride, unsigned out_stride)
{
	v4f r0, i0, r1, i1, r2, i2;
	v4f or0, oi0, or1, oi1, or2, oi2;
	v4f ar1, ai1, ar2, ai2;
	v4f cr1, ci1, cr2, ci2;
	v4f dr1, di1, dr2, di2;
	v4f er1, ei1, er2, ei2;
	v4f tr1, ti1, tr2, ti2;
	v4f tr3, ti3, tr4, ti4;
	v4f tr5, ti5;
	V4F_LD2(r0, i0, in + 0*in_stride);
	V4F_LD2(r1, i1, in + 1*in_stride);
	V4F_LD2(r2, i2, in + 2*in_stride);
	cr1 = v4f_broadcast(twid[0]);
	ci1 = v4f_broadcast(twid[1]);
	cr2 = v4f_broadcast(twid[2]);
	ci2 = v4f_broadcast(twid[3]);
	dr1 = v4f_mul(r1, cr1);
	er1 = v4f_mul(i1, ci1);
	di1 = v4f_mul(r1, ci1);
	ei1 = v4f_mul(i1, cr1);
	dr2 = v4f_mul(r2, cr2);
	er2 = v4f_mul(i2, ci2);
	di2 = v4f_mul(r2, ci2);
	ei2 = v4f_mul(i2, cr2);
	ar1 = v4f_sub(dr1, er1);
	ai1 = v4f_add(di1, ei1);
	ar2 = v4f_sub(dr2, er2);
	ai2 = v4f_add(di2, ei2);
	tr1 = v4f_add(ar2, ar1);
	ti1 = v4f_add(ai2, ai1);
	tr2 = v4f_sub(ai2, ai1);
	ti2 = v4f_sub(ar2, ar1);
	tr5 = v4f_mul(tr1, C_C3);
	ti5 = v4f_mul(ti1, C_C3);
	tr3 = v4f_mul(tr2, C_S3);
	ti3 = v4f_mul(ti2, C_S3);
	tr4 = v4f_sub(r0, tr5);
	ti4 = v4f_sub(i0, ti5);
	or0 = v4f_add(r0, tr1);
	oi0 = v4f_add(i0, ti1);
	or1 = v4f_sub(tr4, tr3);
	oi1 = v4f_add(ti4, ti3);
	or2 = v4f_add(tr4, tr3);
	oi2 = v4f_sub(ti4, ti3);
	V4F_ST2(out + 0*out_stride, or0, oi0);
	V4F_ST2(out + 1*out_stride, or1, oi1);
	V4F_ST2(out + 2*out_stride, or2, oi2);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dif_fft4_offset_io(const float *in, float *out, const float *twid, unsigned in_stride, unsigned out_stride)
{
	v4f b0r   = v4f_ld(in + 0*in_stride + 0);
	v4f b0i   = v4f_ld(in + 0*in_stride + 4);
	v4f b1r   = v4f_ld(in + 1*in_stride + 0);
	v4f b1i   = v4f_ld(in + 1*in_stride + 4);
	v4f b2r   = v4f_ld(in + 2*in_stride + 0);
	v4f b2i   = v4f_ld(in + 2*in_stride + 4);
	v4f b3r   = v4f_ld(in + 3*in_stride + 0);
	v4f b3i   = v4f_ld(in + 3*in_stride + 4);
	v4f yr0   = v4f_add(b0r, b2r);
	v4f yi0   = v4f_add(b0i, b2i);
	v4f yr2   = v4f_sub(b0r, b2r);
	v4f yi2   = v4f_sub(b0i, b2i);
	v4f yr1   = v4f_add(b1r, b3r);
	v4f yi1   = v4f_add(b1i, b3i);
	v4f yr3   = v4f_sub(b1r, b3r);
	v4f yi3   = v4f_sub(b1i, b3i);
	v4f tr0   = v4f_add(yr0, yr1);
	v4f ti0   = v4f_add(yi0, yi1);
	v4f tr2   = v4f_sub(yr0, yr1);
	v4f ti2   = v4f_sub(yi0, yi1);
	v4f tr1   = v4f_add(yr2, yi3);
	v4f ti1   = v4f_sub(yi2, yr3);
	v4f tr3   = v4f_sub(yr2, yi3);
	v4f ti3   = v4f_add(yi2, yr3);
	v4f c1r   = v4f_broadcast(twid[0]);
	v4f c1i   = v4f_broadcast(twid[1]);
	v4f c2r   = v4f_broadcast(twid[2]);
	v4f c2i   = v4f_broadcast(twid[3]);
	v4f c3r   = v4f_broadcast(twid[4]);
	v4f c3i   = v4f_broadcast(twid[5]);
	v4f o1ra  = v4f_mul(tr1, c1r);
	v4f o1rb  = v4f_mul(ti1, c1i);
	v4f o1ia  = v4f_mul(tr1, c1i);
	v4f o1ib  = v4f_mul(ti1, c1r);
	v4f o2ra  = v4f_mul(tr2, c2r);
	v4f o2rb  = v4f_mul(ti2, c2i);
	v4f o2ia  = v4f_mul(tr2, c2i);
	v4f o2ib  = v4f_mul(ti2, c2r);
	v4f o3ra  = v4f_mul(tr3, c3r);
	v4f o3rb  = v4f_mul(ti3, c3i);
	v4f o3ia  = v4f_mul(tr3, c3i);
	v4f o3ib  = v4f_mul(ti3, c3r);
	v4f o1r   = v4f_sub(o1ra, o1rb);
	v4f o1i   = v4f_add(o1ia, o1ib);
	v4f o2r   = v4f_sub(o2ra, o2rb);
	v4f o2i   = v4f_add(o2ia, o2ib);
	v4f o3r   = v4f_sub(o3ra, o3rb);
	v4f o3i   = v4f_add(o3ia, o3ib);
	v4f_st(out + 0*out_stride + 0, tr0);
	v4f_st(out + 0*out_stride + 4, ti0);
	v4f_st(out + 1*out_stride + 0, o1r);
	v4f_st(out + 1*out_stride + 4, o1i);
	v4f_st(out + 2*out_stride + 0, o2r);
	v4f_st(out + 2*out_stride + 4, o2i);
	v4f_st(out + 3*out_stride + 0, o3r);
	v4f_st(out + 3*out_stride + 4, o3i);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dif_fft4_offset_o(const float *in, float *out, unsigned outoffset)
{
	v4f b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i;
	v4f y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
	v4f z0r, z0i, z1r, z1i, z2r, z2i, z3r, z3i;

	V4F_LD2(b0r, b0i, in + 0);
	V4F_LD2(b1r, b1i, in + 8);
	V4F_LD2(b2r, b2i, in + 16);
	V4F_LD2(b3r, b3i, in + 24);
	y0r  = v4f_add(b0r, b2r);
	y0i  = v4f_add(b0i, b2i);
	y2r  = v4f_sub(b0r, b2r);
	y2i  = v4f_sub(b0i, b2i);
	y1r  = v4f_add(b1r, b3r);
	y1i  = v4f_add(b1i, b3i);
	y3r  = v4f_sub(b1r, b3r);
	y3i  = v4f_sub(b1i, b3i);
	z0r  = v4f_add(y0r, y1r);
	z0i  = v4f_add(y0i, y1i);
	z2r  = v4f_sub(y0r, y1r);
	z2i  = v4f_sub(y0i, y1i);
	z1r  = v4f_add(y2r, y3i);
	z1i  = v4f_sub(y2i, y3r);
	z3r  = v4f_sub(y2r, y3i);
	z3i  = v4f_add(y2i, y3r);
	V4F_ST2(out + 0*outoffset, z0r, z0i);
	V4F_ST2(out + 1*outoffset, z1r, z1i);
	V4F_ST2(out + 2*outoffset, z2r, z2i);
	V4F_ST2(out + 3*outoffset, z3r, z3i);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dit_fft4_offset_io(const float *in, float *out, const float *twid, unsigned in_stride, unsigned out_stride)
{
	v4f b0r  = v4f_ld(in + 0*in_stride + 0);
	v4f b0i  = v4f_ld(in + 0*in_stride + 4);
	v4f b1r  = v4f_ld(in + 1*in_stride + 0);
	v4f b1i  = v4f_ld(in + 1*in_stride + 4);
	v4f b2r  = v4f_ld(in + 2*in_stride + 0);
	v4f b2i  = v4f_ld(in + 2*in_stride + 4);
	v4f b3r  = v4f_ld(in + 3*in_stride + 0);
	v4f b3i  = v4f_ld(in + 3*in_stride + 4);
	v4f c1r  = v4f_broadcast(twid[0]);
	v4f c1i  = v4f_broadcast(twid[1]);
	v4f c2r  = v4f_broadcast(twid[2]);
	v4f c2i  = v4f_broadcast(twid[3]);
	v4f c3r  = v4f_broadcast(twid[4]);
	v4f c3i  = v4f_broadcast(twid[5]);
	v4f x1ra = v4f_mul(b1r, c1r);
	v4f x1rb = v4f_mul(b1i, c1i);
	v4f x1ia = v4f_mul(b1r, c1i);
	v4f x1ib = v4f_mul(b1i, c1r);
	v4f x2ra = v4f_mul(b2r, c2r);
	v4f x2rb = v4f_mul(b2i, c2i);
	v4f x2ia = v4f_mul(b2r, c2i);
	v4f x2ib = v4f_mul(b2i, c2r);
	v4f x3ra = v4f_mul(b3r, c3r);
	v4f x3rb = v4f_mul(b3i, c3i);
	v4f x3ia = v4f_mul(b3r, c3i);
	v4f x3ib = v4f_mul(b3i, c3r);
	v4f x1r  = v4f_sub(x1ra, x1rb);
	v4f x1i  = v4f_add(x1ia, x1ib);
	v4f x2r  = v4f_sub(x2ra, x2rb);
	v4f x2i  = v4f_add(x2ia, x2ib);
	v4f x3r  = v4f_sub(x3ra, x3rb);
	v4f x3i  = v4f_add(x3ia, x3ib);
	v4f yr0  = v4f_add(b0r, x2r);
	v4f yi0  = v4f_add(b0i, x2i);
	v4f yr2  = v4f_sub(b0r, x2r);
	v4f yi2  = v4f_sub(b0i, x2i);
	v4f yr1  = v4f_add(x1r, x3r);
	v4f yi1  = v4f_add(x1i, x3i);
	v4f yr3  = v4f_sub(x1r, x3r);
	v4f yi3  = v4f_sub(x1i, x3i);
	v4f o0r  = v4f_add(yr0, yr1);
	v4f o0i  = v4f_add(yi0, yi1);
	v4f o2r  = v4f_sub(yr0, yr1);
	v4f o2i  = v4f_sub(yi0, yi1);
	v4f o1r  = v4f_add(yr2, yi3);
	v4f o1i  = v4f_sub(yi2, yr3);
	v4f o3r  = v4f_sub(yr2, yi3);
	v4f o3i  = v4f_add(yi2, yr3);
	v4f_st(out + 0*out_stride + 0, o0r);
	v4f_st(out + 0*out_stride + 4, o0i);
	v4f_st(out + 1*out_stride + 0, o1r);
	v4f_st(out + 1*out_stride + 4, o1i);
	v4f_st(out + 2*out_stride + 0, o2r);
	v4f_st(out + 2*out_stride + 4, o2i);
	v4f_st(out + 3*out_stride + 0, o3r);
	v4f_st(out + 3*out_stride + 4, o3i);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dif_fft8_offset_o(const float *in, float *out, unsigned outoffset)
{
	static const float root_half = -0.707106781186548f;
	const v4f vec_root_half = v4f_broadcast(root_half);
	v4f a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, a4r, a4i, a5r, a5i, a6r, a6i, a7r, a7i;
	v4f b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i, b4r, b4i, b5r, b5i, b6r, b6i, b7r, b7i;
	v4f c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i, c4r, c4i, c5r, c5i, c6r, c6i, c7r, c7i;
	v4f d0r, d0i, d1r, d1i, d2r, d2i, d3r, d3i, d4r, d4i, d5r, d5i, d6r, d6i, d7r, d7i;
	v4f e0r, e0i, e2r, e2i, e3r, e3i, e4r, e4i;

	V4F_LD2(a0r, a0i, in + 0);
	V4F_LD2(a1r, a1i, in + 8);
	V4F_LD2(a2r, a2i, in + 16);
	V4F_LD2(a3r, a3i, in + 24);
	V4F_LD2(a4r, a4i, in + 32);
	V4F_LD2(a5r, a5i, in + 40);
	V4F_LD2(a6r, a6i, in + 48);
	V4F_LD2(a7r, a7i, in + 56);

	b0r = v4f_add(a0r, a4r);
	b4r = v4f_sub(a0r, a4r);
	b0i = v4f_add(a0i, a4i);
	b4i = v4f_sub(a0i, a4i);

	b1r = v4f_add(a1r, a5r);
	e0r = v4f_sub(a1r, a5r);
	b1i = v4f_add(a1i, a5i);
	e0i = v4f_sub(a5i, a1i);

	b2r = v4f_add(a2r, a6r);
	b6i = v4f_sub(a6r, a2r);
	b2i = v4f_add(a2i, a6i);
	b6r = v4f_sub(a2i, a6i);

	b3r = v4f_add(a3r, a7r);
	e2r = v4f_sub(a3r, a7r);
	b3i = v4f_add(a3i, a7i);
	e2i = v4f_sub(a3i, a7i);

	c0r = v4f_add(b0r, b2r);
	c2r = v4f_sub(b0r, b2r);
	c0i = v4f_add(b0i, b2i);
	c2i = v4f_sub(b0i, b2i);

	c1r = v4f_add(b1r, b3r);
	c3r = v4f_sub(b1r, b3r);
	c1i = v4f_add(b1i, b3i);
	c3i = v4f_sub(b1i, b3i);

	d0r = v4f_add(c0r, c1r);
	d4r = v4f_sub(c0r, c1r);
	d0i = v4f_add(c0i, c1i);
	d4i = v4f_sub(c0i, c1i);

	V4F_ST2(out + 0*outoffset, d0r, d0i);

	d2r = v4f_add(c2r, c3i);
	d6r = v4f_sub(c2r, c3i);
	d2i = v4f_sub(c2i, c3r);
	d6i = v4f_add(c2i, c3r);

	e3r = v4f_sub(e0i, e0r);
	e3i = v4f_add(e0r, e0i);
	e4r = v4f_sub(e2r, e2i);
	e4i = v4f_add(e2r, e2i);

	b5r = v4f_mul(e3r, vec_root_half);
	b5i = v4f_mul(e3i, vec_root_half);
	b7r = v4f_mul(e4r, vec_root_half);
	b7i = v4f_mul(e4i, vec_root_half);

	c4r = v4f_add(b4r, b6r);
	c6r = v4f_sub(b4r, b6r);
	c4i = v4f_add(b4i, b6i);
	c6i = v4f_sub(b4i, b6i);

	c5r = v4f_add(b5r, b7r);
	c7r = v4f_sub(b5r, b7r);
	c5i = v4f_add(b5i, b7i);
	c7i = v4f_sub(b5i, b7i);

	d1r = v4f_add(c4r, c5r);
	d5r = v4f_sub(c4r, c5r);
	d1i = v4f_add(c4i, c5i);
	d5i = v4f_sub(c4i, c5i);

	d3r = v4f_add(c6r, c7i);
	d7r = v4f_sub(c6r, c7i);
	d3i = v4f_sub(c6i, c7r);
	d7i = v4f_add(c6i, c7r);

	V4F_ST2(out + 1*outoffset, d1r, d1i);
	V4F_ST2(out + 2*outoffset, d2r, d2i);
	V4F_ST2(out + 3*outoffset, d3r, d3i);
	V4F_ST2(out + 4*outoffset, d4r, d4i);
	V4F_ST2(out + 5*outoffset, d5r, d5i);
	V4F_ST2(out + 6*outoffset, d6r, d6i);
	V4F_ST2(out + 7*outoffset, d7r, d7i);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dif_fft16_offset_o(const float *in, float *out, unsigned outoffset)
{
	const v4f VC_C4 = v4f_broadcast(C_C4);
	const v4f VC_C8 = v4f_broadcast(C_C8);
	const v4f VC_S8 = v4f_broadcast(C_S8);

	float VEC_ALIGN_BEST stack[4*2*16];

	v4f a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i;
	v4f b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i;
	v4f c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i;
	v4f d0r, d0i, d1r, d1i, d2r, d2i, d3r, d3i;
	v4f y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
	v4f z0r, z0i, z1r, z1i, z2r, z2i, z3r, z3i;
	v4f e1r, e1i, e2r, e2i, e3r, e3i;

	V4F_LD2(a0r, a0i, in + 0);
	V4F_LD2(b0r, b0i, in + 32);
	V4F_LD2(c0r, c0i, in + 64);
	V4F_LD2(d0r, d0i, in + 96);

	/* fft 4 */
	y0r  = v4f_add(a0r, c0r);
	y0i  = v4f_add(a0i, c0i);
	y2r  = v4f_sub(a0r, c0r);
	y2i  = v4f_sub(a0i, c0i);
	y1r  = v4f_add(b0r, d0r);
	y1i  = v4f_add(b0i, d0i);
	y3r  = v4f_sub(b0r, d0r);
	y3i  = v4f_sub(b0i, d0i);
	z0r  = v4f_add(y0r, y1r);
	z0i  = v4f_add(y0i, y1i);
	z2r  = v4f_sub(y0r, y1r);
	z2i  = v4f_sub(y0i, y1i);
	z1r  = v4f_add(y2r, y3i);
	z1i  = v4f_sub(y2i, y3r);
	z3r  = v4f_sub(y2r, y3i);
	z3i  = v4f_add(y2i, y3r);

	V4F_ST2(stack + 0,  z0r, z0i);
	V4F_LD2(a1r, a1i, in + 8);
	V4F_ST2(stack + 32, z1r, z1i);
	V4F_LD2(b1r, b1i, in + 40);
	V4F_ST2(stack + 64, z2r, z2i);
	V4F_LD2(c1r, c1i, in + 72);
	V4F_ST2(stack + 96, z3r, z3i);
	V4F_LD2(d1r, d1i, in + 104);

	/* fft 4 */
	y0r  = v4f_add(a1r, c1r);
	y2r  = v4f_sub(a1r, c1r);
	y0i  = v4f_add(a1i, c1i);
	y2i  = v4f_sub(a1i, c1i);
	y1r  = v4f_add(b1r, d1r);
	y3r  = v4f_sub(b1r, d1r);
	y1i  = v4f_add(b1i, d1i);
	y3i  = v4f_sub(b1i, d1i);
	z0r  = v4f_add(y0r, y1r);
	z2r  = v4f_sub(y0r, y1r);
	z0i  = v4f_add(y0i, y1i);
	z2i  = v4f_sub(y0i, y1i);
	z1r  = v4f_add(y2r, y3i);
	z3r  = v4f_sub(y2r, y3i);
	z1i  = v4f_sub(y2i, y3r);
	z3i  = v4f_add(y2i, y3r);

	/* twiddle */
	e1r  = v4f_mul(z1r, VC_C8);
	e1i  = v4f_mul(z1i, VC_C8);
	z1r  = v4f_mul(z1r, VC_S8);
	z1i  = v4f_mul(z1i, VC_S8);
	e1r  = v4f_add(e1r, z1i);
	e1i  = v4f_sub(e1i, z1r);
	z2r  = v4f_mul(z2r, VC_C4);
	z2i  = v4f_mul(z2i, VC_C4);
	e2r  = v4f_add(z2r, z2i);
	e2i  = v4f_sub(z2i, z2r);
	e3r  = v4f_mul(z3r, VC_S8);
	e3i  = v4f_mul(z3i, VC_S8);
	z3i  = v4f_mul(z3i, VC_C8);
	z3r  = v4f_mul(z3r, VC_C8);
	e3r  = v4f_add(e3r, z3i);
	e3i  = v4f_sub(e3i, z3r);

	V4F_ST2(stack + 8,   z0r, z0i);
	V4F_LD2(a2r, a2i, in + 16);
	V4F_ST2(stack + 40,  e1r, e1i);
	V4F_LD2(b2r, b2i, in + 48);
	V4F_ST2(stack + 72,  e2r, e2i);
	V4F_LD2(c2r, c2i, in + 80);
	V4F_ST2(stack + 104, e3r, e3i);
	V4F_LD2(d2r, d2i, in + 112);

	/* fft 4 */
	y0r  = v4f_add(a2r, c2r);
	y2r  = v4f_sub(a2r, c2r);
	y0i  = v4f_add(a2i, c2i);
	y2i  = v4f_sub(a2i, c2i);
	y1r  = v4f_add(b2r, d2r);
	y3r  = v4f_sub(b2r, d2r);
	y1i  = v4f_add(b2i, d2i);
	y3i  = v4f_sub(b2i, d2i);
	z0r  = v4f_add(y1r, y0r);
	z2r  = v4f_sub(y1r, y0r);
	z0i  = v4f_add(y0i, y1i);
	z2i  = v4f_sub(y0i, y1i);
	z1r  = v4f_add(y3i, y2r);
	z3r  = v4f_sub(y3i, y2r);
	z1i  = v4f_sub(y2i, y3r);
	z3i  = v4f_add(y2i, y3r);

	/* twiddle */
	e1r  = v4f_add(z1i, z1r);
	e1i  = v4f_sub(z1i, z1r);
	e3r  = v4f_add(z3r, z3i);
	e3i  = v4f_sub(z3r, z3i);
	e1r  = v4f_mul(e1r, VC_C4);
	e1i  = v4f_mul(e1i, VC_C4);
	e3r  = v4f_mul(e3r, VC_C4);
	e3i  = v4f_mul(e3i, VC_C4);

	V4F_ST2(stack + 16,  z0r, z0i);
	V4F_LD2(a3r, a3i, in + 24);
	V4F_ST2(stack + 48,  e1r, e1i);
	V4F_LD2(b3r, b3i, in + 56);
	V4F_ST2(stack + 80,  z2i, z2r);
	V4F_LD2(c3r, c3i, in + 88);
	V4F_ST2(stack + 112, e3r, e3i);
	V4F_LD2(d3r, d3i, in + 120);

	/* fft 4 */
	y0r  = v4f_add(a3r, c3r);
	y2r  = v4f_sub(a3r, c3r);
	y0i  = v4f_add(a3i, c3i);
	y2i  = v4f_sub(a3i, c3i);
	y1r  = v4f_add(b3r, d3r);
	y3r  = v4f_sub(d3r, b3r);
	y1i  = v4f_add(b3i, d3i);
	y3i  = v4f_sub(b3i, d3i);
	z0r  = v4f_add(y1r, y0r);
	z2r  = v4f_sub(y1r, y0r);
	z0i  = v4f_add(y0i, y1i);
	z2i  = v4f_sub(y0i, y1i);
	z1r  = v4f_add(y2r, y3i);
	z3r  = v4f_sub(y2r, y3i);
	z1i  = v4f_add(y3r, y2i);
	z3i  = v4f_sub(y3r, y2i);

	/* twiddle */
	e1r  = v4f_mul(z1r, VC_S8);
	e1i  = v4f_mul(z1i, VC_S8);
	z1i  = v4f_mul(z1i, VC_C8);
	z1r  = v4f_mul(z1r, VC_C8);
	e1r  = v4f_add(e1r, z1i);
	e1i  = v4f_sub(e1i, z1r);
	e2r  = v4f_add(z2r, z2i);
	e2i  = v4f_sub(z2r, z2i);
	e2r  = v4f_mul(e2r, VC_C4);
	e2i  = v4f_mul(e2i, VC_C4);
	e3r  = v4f_mul(z3i, VC_S8);
	e3i  = v4f_mul(z3r, VC_S8);
	z3r  = v4f_mul(z3r, VC_C8);
	z3i  = v4f_mul(z3i, VC_C8);
	e3r  = v4f_sub(e3r, z3r);
	e3i  = v4f_add(e3i, z3i);

	V4F_ST2(stack + 24,  z0r, z0i);
	fftset_vec_dif_fft4_offset_o(stack + 0,  out + 0*outoffset, 4*outoffset);

	V4F_ST2(stack + 56,  e1r, e1i);
	fftset_vec_dif_fft4_offset_o(stack + 32, out + 1*outoffset, 4*outoffset);

	V4F_ST2(stack + 88,  e2r, e2i);
	fftset_vec_dif_fft4_offset_o(stack + 64, out + 2*outoffset, 4*outoffset);

	V4F_ST2(stack + 120, e3r, e3i);
	fftset_vec_dif_fft4_offset_o(stack + 96, out + 3*outoffset, 4*outoffset);
}

#define BUILD_INNER_PASSES(n_) \
static void fc_v4_r ## n_ ## _inner(float *work_buf, unsigned nfft, unsigned lfft, const float *twid) \
{ \
	assert(lfft == 1); \
	do { \
		fftset_vec_dif_fft ## n_ ## _offset_o(work_buf, work_buf, 2 * FFT_VEC_LEN); \
		work_buf += 2 * (n_) * FFT_VEC_LEN; \
	} while (--nfft); \
} \
static void fc_v4_stock_r ## n_ ## _inner(const float *in, float *out, const float *twid, unsigned ncol, unsigned nrow_div_radix) \
{ \
	const unsigned ioffset = (2*FFT_VEC_LEN)*nrow_div_radix; \
	assert(ncol == 1); \
	do { \
		fftset_vec_dif_fft ## n_ ## _offset_o(in, out, ioffset); \
		out += 2 * FFT_VEC_LEN; \
		in  += 2 * (n_) * FFT_VEC_LEN; \
	} while (--nrow_div_radix); \
}

#define BUILD_STANDARD_PASSES(n_, ntwid_) \
static void fc_v4_dif_r ## n_(float *work_buf, unsigned nfft, unsigned lfft, const float *twid) \
{ \
	unsigned rinc = lfft * (2 * FFT_VEC_LEN); \
	do { \
		unsigned j; \
		const float *tp = twid; \
		for (j = 0; j < lfft; j++, work_buf += 2 * FFT_VEC_LEN, tp += (ntwid_)) { \
			fftset_vec_dif_fft ## n_ ## _offset_io(work_buf, work_buf, tp, rinc, rinc); \
		} \
		work_buf += ((n_) - 1)*rinc; \
	} while (--nfft); \
} \
static void fc_v4_dit_r ## n_(float *work_buf, unsigned nfft, unsigned lfft, const float *twid) \
{ \
	unsigned rinc = lfft * (2 * FFT_VEC_LEN); \
	do { \
		unsigned j; \
		const float *tp = twid; \
		for (j = 0; j < lfft; j++, work_buf += 2 * FFT_VEC_LEN, tp += (ntwid_)) { \
			fftset_vec_dit_fft ## n_ ## _offset_io(work_buf, work_buf, tp, rinc, rinc); \
		} \
		work_buf += ((n_) - 1)*rinc; \
	} while (--nfft); \
} \
static void fc_v4_stock_r ## n_(const float *in, float *out, const float *twid, unsigned ncol, unsigned nrow_div_radix) \
{ \
	const unsigned ooffset = (2*FFT_VEC_LEN)*ncol; \
	const unsigned ioffset = ooffset*nrow_div_radix; \
	do { \
		const float *in0 = in; \
		const float *tp  = twid; \
		unsigned     j   = ncol; \
		do { \
			fftset_vec_dif_fft ## n_ ## _offset_io(in0, out, tp, ooffset, ioffset); \
			tp   += (ntwid_); \
			out  += (2*FFT_VEC_LEN); \
			in0  += (2*FFT_VEC_LEN); \
		} while (--j); \
		in = in + (n_)*ooffset; \
	} while (--nrow_div_radix); \
} \
BUILD_INNER_PASSES(n_)

BUILD_STANDARD_PASSES(2, 2)
BUILD_STANDARD_PASSES(3, 4)
BUILD_STANDARD_PASSES(4, 6)
BUILD_INNER_PASSES(8)
BUILD_INNER_PASSES(16)

struct fftset_vec *fastconv_get_inner_pass(struct fftset *fc, unsigned length)
{
	struct fftset_vec *pass;
	struct fftset_vec **ipos;
	unsigned i;

	/* Search for the pass. */
	for (pass = fc->first_inner; pass != NULL; pass = pass->next) {
		if (pass->lfft_div_radix*pass->radix == length && pass->dif != NULL)
			return pass;
		if (pass->lfft_div_radix*pass->radix < length)
			break;
	}

	/* Create new inner pass. */
	pass = aalloc_alloc(&fc->memory, sizeof(*pass));
	if (pass == NULL)
		return NULL;

	/* Detect radix. */
	if (length == 2) {
		pass->twiddle        = NULL;
		pass->lfft_div_radix = 1;
		pass->radix          = 2;
		pass->dif            = fc_v4_r2_inner;
		pass->dit            = fc_v4_r2_inner;
		pass->dif_stockham   = fc_v4_stock_r2_inner;
	} else if (length == 3) {
		pass->twiddle        = NULL;
		pass->lfft_div_radix = 1;
		pass->radix          = 3;
		pass->dif            = fc_v4_r3_inner;
		pass->dit            = fc_v4_r3_inner;
		pass->dif_stockham   = fc_v4_stock_r3_inner;
	} else if (length == 4) {
		pass->twiddle        = NULL;
		pass->lfft_div_radix = 1;
		pass->radix          = 4;
		pass->dif            = fc_v4_r4_inner;
		pass->dit            = fc_v4_r4_inner;
		pass->dif_stockham   = fc_v4_stock_r4_inner;
	} else if (length == 8) {
		pass->twiddle        = NULL;
		pass->lfft_div_radix = 1;
		pass->radix          = 8;
		pass->dif            = fc_v4_r8_inner;
		pass->dit            = fc_v4_r8_inner;
		pass->dif_stockham   = fc_v4_stock_r8_inner;
	} else if (length == 16) {
		pass->twiddle        = NULL;
		pass->lfft_div_radix = 1;
		pass->radix          = 16;
		pass->dif            = fc_v4_r16_inner;
		pass->dit            = fc_v4_r16_inner;
		pass->dif_stockham   = fc_v4_stock_r16_inner;
	} else if (length % 3 == 0) {
		float *twid = aalloc_align_alloc(&fc->memory, sizeof(float) * 4 * length / 3, 64);
		if (twid == NULL)
			return NULL;
		for (i = 0; i < length / 3; i++) {
			twid[4*i+0] = cosf(i * (-(float)M_PI * 2.0f) / length);
			twid[4*i+1] = sinf(i * (-(float)M_PI * 2.0f) / length);
			twid[4*i+2] = cosf(i * (-(float)M_PI * 4.0f) / length);
			twid[4*i+3] = sinf(i * (-(float)M_PI * 4.0f) / length);
		}
		pass->twiddle        = twid;
		pass->lfft_div_radix = length / 3;
		pass->radix          = 3;
		pass->dif            = fc_v4_dif_r3;
		pass->dit            = fc_v4_dit_r3;
		pass->dif_stockham   = fc_v4_stock_r3;
	} else if (length % 4 == 0 && length / 4 != 8 && length / 4 != 4) {
		float *twid = aalloc_align_alloc(&fc->memory, sizeof(float) * 6 * length / 4, 64);
		if (twid == NULL)
			return NULL;
		for (i = 0; i < length / 4; i++) {
			twid[6*i+0] = cosf(i * (-(float)M_PI * 2.0f) / length);
			twid[6*i+1] = sinf(i * (-(float)M_PI * 2.0f) / length);
			twid[6*i+2] = cosf(i * (-(float)M_PI * 4.0f) / length);
			twid[6*i+3] = sinf(i * (-(float)M_PI * 4.0f) / length);
			twid[6*i+4] = cosf(i * (-(float)M_PI * 6.0f) / length);
			twid[6*i+5] = sinf(i * (-(float)M_PI * 6.0f) / length);
		}
		pass->twiddle      = twid;
		pass->lfft_div_radix = length / 4;
		pass->radix        = 4;
		pass->dif          = fc_v4_dif_r4;
		pass->dit          = fc_v4_dit_r4;
		pass->dif_stockham = fc_v4_stock_r4;
	} else if (length % 2 == 0) {
		float *twid = aalloc_align_alloc(&fc->memory, sizeof(float) * length, 64);
		if (twid == NULL)
			return NULL;
		for (i = 0; i < length / 2; i++) {
			twid[2*i+0] = cosf(i * (-(float)M_PI * 2.0f) / length);
			twid[2*i+1] = sinf(i * (-(float)M_PI * 2.0f) / length);
		}
		pass->twiddle      = twid;
		pass->lfft_div_radix = length / 2;
		pass->radix        = 2;
		pass->dif          = fc_v4_dif_r2;
		pass->dit          = fc_v4_dit_r2;
		pass->dif_stockham = fc_v4_stock_r2;
	} else {
		/* Only support radix-2/3/4. What are you doing crazy-face? */
		abort();
	}

	/* Make next pass if required */
	if (pass->lfft_div_radix != 1) {
		pass->next_compat = fastconv_get_inner_pass(fc, pass->lfft_div_radix);
		if (pass->next_compat == NULL)
			return NULL;
	} else {
		pass->next_compat = NULL;
	}

	/* Insert into list. */
	ipos = &(fc->first_inner);
	while (*ipos != NULL && length < (*ipos)->lfft_div_radix * (*ipos)->radix) {
		ipos = &(*ipos)->next;
	}
	pass->next = *ipos;
	*ipos = pass;

	return pass;
}
