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

static const float C_C3 = 0.5f;               /* PI / 3 */
static const float C_S3 = 0.86602540378444f;  /* PI / 3 */
static const float C_C4 = 0.707106781186548f; /* PI / 4 */
static const float C_C8 = 0.923879532511288f; /* PI / 8 */
static const float C_S8 = 0.382683432365086f; /* PI / 8 */

static COP_ATTR_ALWAYSINLINE void fftset_vec_dif_fft2_offset_io(const float *in, float *out, const float *twid, unsigned in_stride, unsigned out_stride)
{
	vlf nre   = vlf_ld(in + 0*VLF_WIDTH);
	vlf nim   = vlf_ld(in + 1*VLF_WIDTH);
	vlf fre   = vlf_ld(in + in_stride + 0*VLF_WIDTH);
	vlf fim   = vlf_ld(in + in_stride + 1*VLF_WIDTH);
	vlf tre   = vlf_broadcast(twid[0]);
	vlf tim   = vlf_broadcast(twid[1]);
	vlf onre  = vlf_add(nre, fre);
	vlf onim  = vlf_add(nim, fim);
	vlf ptre  = vlf_sub(nre, fre);
	vlf ptim  = vlf_sub(nim, fim);
	vlf ofrea = vlf_mul(ptre, tre);
	vlf ofreb = vlf_mul(ptim, tim);
	vlf ofima = vlf_mul(ptre, tim);
	vlf ofimb = vlf_mul(ptim, tre);
	vlf ofre  = vlf_sub(ofrea, ofreb);
	vlf ofim  = vlf_add(ofima, ofimb);
	vlf_st(out + 0*VLF_WIDTH, onre);
	vlf_st(out + 1*VLF_WIDTH, onim);
	vlf_st(out + out_stride + 0*VLF_WIDTH, ofre);
	vlf_st(out + out_stride + 1*VLF_WIDTH, ofim);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dif_fft2_offset_o(const float *in, float *out, unsigned out_stride)
{
	vlf nre, nim, fre, fim;
	vlf onre, onim, ofre, ofim;
	VLF_LD2(nre, nim, in + 0*VLF_WIDTH);
	VLF_LD2(fre, fim, in + 2*VLF_WIDTH);
	onre = vlf_add(nre, fre);
	onim = vlf_add(nim, fim);
	ofre = vlf_sub(nre, fre);
	ofim = vlf_sub(nim, fim);
	VLF_ST2(out + 0, onre, onim);
	VLF_ST2(out + out_stride, ofre, ofim);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dit_fft2_offset_io(const float *in, float *out, const float *twid, unsigned in_stride, unsigned out_stride)
{
	vlf nre  = vlf_ld(in + 0*VLF_WIDTH);
	vlf nim  = vlf_ld(in + 1*VLF_WIDTH);
	vlf ptre = vlf_ld(in + in_stride + 0*VLF_WIDTH);
	vlf ptim = vlf_ld(in + in_stride + 1*VLF_WIDTH);
	vlf tre  = vlf_broadcast(twid[0]);
	vlf tim  = vlf_broadcast(twid[1]);
	vlf frea = vlf_mul(ptre, tre);
	vlf freb = vlf_mul(ptim, tim);
	vlf fima = vlf_mul(ptre, tim);
	vlf fimb = vlf_mul(ptim, tre);
	vlf fre  = vlf_sub(frea, freb);
	vlf fim  = vlf_add(fima, fimb);
	vlf onre = vlf_add(nre, fre);
	vlf onim = vlf_add(nim, fim);
	vlf ofre = vlf_sub(nre, fre);
	vlf ofim = vlf_sub(nim, fim);
	vlf_st(out + 0*VLF_WIDTH, onre);
	vlf_st(out + 1*VLF_WIDTH, onim);
	vlf_st(out + out_stride + 0*VLF_WIDTH, ofre);
	vlf_st(out + out_stride + 1*VLF_WIDTH, ofim);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dif_fft3_offset_io(const float *in, float *out, const float *twid, unsigned in_stride, unsigned out_stride)
{
	vlf r0, i0, r1, i1, r2, i2;
	vlf or0, oi0, or1, oi1, or2, oi2;
	vlf ar1, ai1, ar2, ai2;
	vlf cr1, ci1, cr2, ci2;
	vlf dr1, di1, dr2, di2;
	vlf er1, ei1, er2, ei2;
	vlf tr1, ti1, tr2, ti2;
	vlf tr3, ti3, tr4, ti4;
	vlf tr5, ti5;
	const vlf coef0 = vlf_broadcast(C_C3);
	const vlf coef1 = vlf_broadcast(C_S3);
	VLF_LD2(r0, i0, in + 0*in_stride);
	VLF_LD2(r1, i1, in + 1*in_stride);
	VLF_LD2(r2, i2, in + 2*in_stride);
	tr1 = vlf_add(r2, r1);
	ti1 = vlf_add(i2, i1);
	tr2 = vlf_sub(i2, i1);
	ti2 = vlf_sub(r2, r1);
	tr5 = vlf_mul(tr1, coef0);
	ti5 = vlf_mul(ti1, coef0);
	tr3 = vlf_mul(tr2, coef1);
	ti3 = vlf_mul(ti2, coef1);
	tr4 = vlf_sub(r0, tr5);
	ti4 = vlf_sub(i0, ti5);
	or0 = vlf_add(r0, tr1);
	oi0 = vlf_add(i0, ti1);
	ar1 = vlf_sub(tr4, tr3);
	ai1 = vlf_add(ti4, ti3);
	ar2 = vlf_add(tr4, tr3);
	ai2 = vlf_sub(ti4, ti3);
	cr1 = vlf_broadcast(twid[0]);
	ci1 = vlf_broadcast(twid[1]);
	cr2 = vlf_broadcast(twid[2]);
	ci2 = vlf_broadcast(twid[3]);
	dr1 = vlf_mul(ar1, cr1);
	er1 = vlf_mul(ai1, ci1);
	di1 = vlf_mul(ar1, ci1);
	ei1 = vlf_mul(ai1, cr1);
	dr2 = vlf_mul(ar2, cr2);
	er2 = vlf_mul(ai2, ci2);
	di2 = vlf_mul(ar2, ci2);
	ei2 = vlf_mul(ai2, cr2);
	or1 = vlf_sub(dr1, er1);
	oi1 = vlf_add(di1, ei1);
	or2 = vlf_sub(dr2, er2);
	oi2 = vlf_add(di2, ei2);
	VLF_ST2(out + 0*out_stride, or0, oi0);
	VLF_ST2(out + 1*out_stride, or1, oi1);
	VLF_ST2(out + 2*out_stride, or2, oi2);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dif_fft3_offset_o(const float *in, float *out, unsigned out_stride)
{
	vlf r0, i0, r1, i1, r2, i2;
	vlf or0, oi0, or1, oi1, or2, oi2;
	vlf tr1, ti1, tr2, ti2;
	vlf tr3, ti3, tr4, ti4;
	vlf tr5, ti5;
	const vlf coef0 = vlf_broadcast(C_C3);
	const vlf coef1 = vlf_broadcast(C_S3);
	VLF_LD2(r0, i0, in + 0*VLF_WIDTH);
	VLF_LD2(r1, i1, in + 2*VLF_WIDTH);
	VLF_LD2(r2, i2, in + 4*VLF_WIDTH);
	tr1 = vlf_add(r2, r1);
	ti1 = vlf_add(i2, i1);
	tr2 = vlf_sub(i2, i1);
	ti2 = vlf_sub(r2, r1);
	tr5 = vlf_mul(tr1, coef0);
	ti5 = vlf_mul(ti1, coef0);
	tr3 = vlf_mul(tr2, coef1);
	ti3 = vlf_mul(ti2, coef1);
	tr4 = vlf_sub(r0, tr5);
	ti4 = vlf_sub(i0, ti5);
	or0 = vlf_add(r0, tr1);
	oi0 = vlf_add(i0, ti1);
	or1 = vlf_sub(tr4, tr3);
	oi1 = vlf_add(ti4, ti3);
	or2 = vlf_add(tr4, tr3);
	oi2 = vlf_sub(ti4, ti3);
	VLF_ST2(out + 0*out_stride, or0, oi0);
	VLF_ST2(out + 1*out_stride, or1, oi1);
	VLF_ST2(out + 2*out_stride, or2, oi2);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dit_fft3_offset_io(const float *in, float *out, const float *twid, unsigned in_stride, unsigned out_stride)
{
	vlf r0, i0, r1, i1, r2, i2;
	vlf or0, oi0, or1, oi1, or2, oi2;
	vlf ar1, ai1, ar2, ai2;
	vlf cr1, ci1, cr2, ci2;
	vlf dr1, di1, dr2, di2;
	vlf er1, ei1, er2, ei2;
	vlf tr1, ti1, tr2, ti2;
	vlf tr3, ti3, tr4, ti4;
	vlf tr5, ti5;
	const vlf coef0 = vlf_broadcast(C_C3);
	const vlf coef1 = vlf_broadcast(C_S3);
	VLF_LD2(r0, i0, in + 0*in_stride);
	VLF_LD2(r1, i1, in + 1*in_stride);
	VLF_LD2(r2, i2, in + 2*in_stride);
	cr1 = vlf_broadcast(twid[0]);
	ci1 = vlf_broadcast(twid[1]);
	cr2 = vlf_broadcast(twid[2]);
	ci2 = vlf_broadcast(twid[3]);
	dr1 = vlf_mul(r1, cr1);
	er1 = vlf_mul(i1, ci1);
	di1 = vlf_mul(r1, ci1);
	ei1 = vlf_mul(i1, cr1);
	dr2 = vlf_mul(r2, cr2);
	er2 = vlf_mul(i2, ci2);
	di2 = vlf_mul(r2, ci2);
	ei2 = vlf_mul(i2, cr2);
	ar1 = vlf_sub(dr1, er1);
	ai1 = vlf_add(di1, ei1);
	ar2 = vlf_sub(dr2, er2);
	ai2 = vlf_add(di2, ei2);
	tr1 = vlf_add(ar2, ar1);
	ti1 = vlf_add(ai2, ai1);
	tr2 = vlf_sub(ai2, ai1);
	ti2 = vlf_sub(ar2, ar1);
	tr5 = vlf_mul(tr1, coef0);
	ti5 = vlf_mul(ti1, coef0);
	tr3 = vlf_mul(tr2, coef1);
	ti3 = vlf_mul(ti2, coef1);
	tr4 = vlf_sub(r0, tr5);
	ti4 = vlf_sub(i0, ti5);
	or0 = vlf_add(r0, tr1);
	oi0 = vlf_add(i0, ti1);
	or1 = vlf_sub(tr4, tr3);
	oi1 = vlf_add(ti4, ti3);
	or2 = vlf_add(tr4, tr3);
	oi2 = vlf_sub(ti4, ti3);
	VLF_ST2(out + 0*out_stride, or0, oi0);
	VLF_ST2(out + 1*out_stride, or1, oi1);
	VLF_ST2(out + 2*out_stride, or2, oi2);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dif_fft4_offset_io(const float *in, float *out, const float *twid, unsigned in_stride, unsigned out_stride)
{
	vlf b0r   = vlf_ld(in + 0*in_stride + 0*VLF_WIDTH);
	vlf b0i   = vlf_ld(in + 0*in_stride + 1*VLF_WIDTH);
	vlf b1r   = vlf_ld(in + 1*in_stride + 0*VLF_WIDTH);
	vlf b1i   = vlf_ld(in + 1*in_stride + 1*VLF_WIDTH);
	vlf b2r   = vlf_ld(in + 2*in_stride + 0*VLF_WIDTH);
	vlf b2i   = vlf_ld(in + 2*in_stride + 1*VLF_WIDTH);
	vlf b3r   = vlf_ld(in + 3*in_stride + 0*VLF_WIDTH);
	vlf b3i   = vlf_ld(in + 3*in_stride + 1*VLF_WIDTH);
	vlf yr0   = vlf_add(b0r, b2r);
	vlf yi0   = vlf_add(b0i, b2i);
	vlf yr2   = vlf_sub(b0r, b2r);
	vlf yi2   = vlf_sub(b0i, b2i);
	vlf yr1   = vlf_add(b1r, b3r);
	vlf yi1   = vlf_add(b1i, b3i);
	vlf yr3   = vlf_sub(b1r, b3r);
	vlf yi3   = vlf_sub(b1i, b3i);
	vlf tr0   = vlf_add(yr0, yr1);
	vlf ti0   = vlf_add(yi0, yi1);
	vlf tr2   = vlf_sub(yr0, yr1);
	vlf ti2   = vlf_sub(yi0, yi1);
	vlf tr1   = vlf_add(yr2, yi3);
	vlf ti1   = vlf_sub(yi2, yr3);
	vlf tr3   = vlf_sub(yr2, yi3);
	vlf ti3   = vlf_add(yi2, yr3);
	vlf c1r   = vlf_broadcast(twid[0]);
	vlf c1i   = vlf_broadcast(twid[1]);
	vlf c2r   = vlf_broadcast(twid[2]);
	vlf c2i   = vlf_broadcast(twid[3]);
	vlf c3r   = vlf_broadcast(twid[4]);
	vlf c3i   = vlf_broadcast(twid[5]);
	vlf o1ra  = vlf_mul(tr1, c1r);
	vlf o1rb  = vlf_mul(ti1, c1i);
	vlf o1ia  = vlf_mul(tr1, c1i);
	vlf o1ib  = vlf_mul(ti1, c1r);
	vlf o2ra  = vlf_mul(tr2, c2r);
	vlf o2rb  = vlf_mul(ti2, c2i);
	vlf o2ia  = vlf_mul(tr2, c2i);
	vlf o2ib  = vlf_mul(ti2, c2r);
	vlf o3ra  = vlf_mul(tr3, c3r);
	vlf o3rb  = vlf_mul(ti3, c3i);
	vlf o3ia  = vlf_mul(tr3, c3i);
	vlf o3ib  = vlf_mul(ti3, c3r);
	vlf o1r   = vlf_sub(o1ra, o1rb);
	vlf o1i   = vlf_add(o1ia, o1ib);
	vlf o2r   = vlf_sub(o2ra, o2rb);
	vlf o2i   = vlf_add(o2ia, o2ib);
	vlf o3r   = vlf_sub(o3ra, o3rb);
	vlf o3i   = vlf_add(o3ia, o3ib);
	vlf_st(out + 0*out_stride + 0*VLF_WIDTH, tr0);
	vlf_st(out + 0*out_stride + 1*VLF_WIDTH, ti0);
	vlf_st(out + 1*out_stride + 0*VLF_WIDTH, o1r);
	vlf_st(out + 1*out_stride + 1*VLF_WIDTH, o1i);
	vlf_st(out + 2*out_stride + 0*VLF_WIDTH, o2r);
	vlf_st(out + 2*out_stride + 1*VLF_WIDTH, o2i);
	vlf_st(out + 3*out_stride + 0*VLF_WIDTH, o3r);
	vlf_st(out + 3*out_stride + 1*VLF_WIDTH, o3i);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dif_fft4_offset_o(const float *in, float *out, unsigned outoffset)
{
	vlf b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i;
	vlf y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
	vlf z0r, z0i, z1r, z1i, z2r, z2i, z3r, z3i;

	VLF_LD2(b0r, b0i, in + 0);
	VLF_LD2(b1r, b1i, in + 8);
	VLF_LD2(b2r, b2i, in + 16);
	VLF_LD2(b3r, b3i, in + 24);
	y0r  = vlf_add(b0r, b2r);
	y0i  = vlf_add(b0i, b2i);
	y2r  = vlf_sub(b0r, b2r);
	y2i  = vlf_sub(b0i, b2i);
	y1r  = vlf_add(b1r, b3r);
	y1i  = vlf_add(b1i, b3i);
	y3r  = vlf_sub(b1r, b3r);
	y3i  = vlf_sub(b1i, b3i);
	z0r  = vlf_add(y0r, y1r);
	z0i  = vlf_add(y0i, y1i);
	z2r  = vlf_sub(y0r, y1r);
	z2i  = vlf_sub(y0i, y1i);
	z1r  = vlf_add(y2r, y3i);
	z1i  = vlf_sub(y2i, y3r);
	z3r  = vlf_sub(y2r, y3i);
	z3i  = vlf_add(y2i, y3r);
	VLF_ST2(out + 0*outoffset, z0r, z0i);
	VLF_ST2(out + 1*outoffset, z1r, z1i);
	VLF_ST2(out + 2*outoffset, z2r, z2i);
	VLF_ST2(out + 3*outoffset, z3r, z3i);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dit_fft4_offset_io(const float *in, float *out, const float *twid, unsigned in_stride, unsigned out_stride)
{
	vlf b0r  = vlf_ld(in + 0*in_stride + 0*VLF_WIDTH);
	vlf b0i  = vlf_ld(in + 0*in_stride + 1*VLF_WIDTH);
	vlf b1r  = vlf_ld(in + 1*in_stride + 0*VLF_WIDTH);
	vlf b1i  = vlf_ld(in + 1*in_stride + 1*VLF_WIDTH);
	vlf b2r  = vlf_ld(in + 2*in_stride + 0*VLF_WIDTH);
	vlf b2i  = vlf_ld(in + 2*in_stride + 1*VLF_WIDTH);
	vlf b3r  = vlf_ld(in + 3*in_stride + 0*VLF_WIDTH);
	vlf b3i  = vlf_ld(in + 3*in_stride + 1*VLF_WIDTH);
	vlf c1r  = vlf_broadcast(twid[0]);
	vlf c1i  = vlf_broadcast(twid[1]);
	vlf c2r  = vlf_broadcast(twid[2]);
	vlf c2i  = vlf_broadcast(twid[3]);
	vlf c3r  = vlf_broadcast(twid[4]);
	vlf c3i  = vlf_broadcast(twid[5]);
	vlf x1ra = vlf_mul(b1r, c1r);
	vlf x1rb = vlf_mul(b1i, c1i);
	vlf x1ia = vlf_mul(b1r, c1i);
	vlf x1ib = vlf_mul(b1i, c1r);
	vlf x2ra = vlf_mul(b2r, c2r);
	vlf x2rb = vlf_mul(b2i, c2i);
	vlf x2ia = vlf_mul(b2r, c2i);
	vlf x2ib = vlf_mul(b2i, c2r);
	vlf x3ra = vlf_mul(b3r, c3r);
	vlf x3rb = vlf_mul(b3i, c3i);
	vlf x3ia = vlf_mul(b3r, c3i);
	vlf x3ib = vlf_mul(b3i, c3r);
	vlf x1r  = vlf_sub(x1ra, x1rb);
	vlf x1i  = vlf_add(x1ia, x1ib);
	vlf x2r  = vlf_sub(x2ra, x2rb);
	vlf x2i  = vlf_add(x2ia, x2ib);
	vlf x3r  = vlf_sub(x3ra, x3rb);
	vlf x3i  = vlf_add(x3ia, x3ib);
	vlf yr0  = vlf_add(b0r, x2r);
	vlf yi0  = vlf_add(b0i, x2i);
	vlf yr2  = vlf_sub(b0r, x2r);
	vlf yi2  = vlf_sub(b0i, x2i);
	vlf yr1  = vlf_add(x1r, x3r);
	vlf yi1  = vlf_add(x1i, x3i);
	vlf yr3  = vlf_sub(x1r, x3r);
	vlf yi3  = vlf_sub(x1i, x3i);
	vlf o0r  = vlf_add(yr0, yr1);
	vlf o0i  = vlf_add(yi0, yi1);
	vlf o2r  = vlf_sub(yr0, yr1);
	vlf o2i  = vlf_sub(yi0, yi1);
	vlf o1r  = vlf_add(yr2, yi3);
	vlf o1i  = vlf_sub(yi2, yr3);
	vlf o3r  = vlf_sub(yr2, yi3);
	vlf o3i  = vlf_add(yi2, yr3);
	vlf_st(out + 0*out_stride + 0*VLF_WIDTH, o0r);
	vlf_st(out + 0*out_stride + 1*VLF_WIDTH, o0i);
	vlf_st(out + 1*out_stride + 0*VLF_WIDTH, o1r);
	vlf_st(out + 1*out_stride + 1*VLF_WIDTH, o1i);
	vlf_st(out + 2*out_stride + 0*VLF_WIDTH, o2r);
	vlf_st(out + 2*out_stride + 1*VLF_WIDTH, o2i);
	vlf_st(out + 3*out_stride + 0*VLF_WIDTH, o3r);
	vlf_st(out + 3*out_stride + 1*VLF_WIDTH, o3i);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dif_fft8_offset_o(const float *in, float *out, unsigned outoffset)
{
	static const float root_half = -0.707106781186548f;
	const vlf vec_root_half = vlf_broadcast(root_half);
	vlf a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, a4r, a4i, a5r, a5i, a6r, a6i, a7r, a7i;
	vlf b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i, b4r, b4i, b5r, b5i, b6r, b6i, b7r, b7i;
	vlf c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i, c4r, c4i, c5r, c5i, c6r, c6i, c7r, c7i;
	vlf d0r, d0i, d1r, d1i, d2r, d2i, d3r, d3i, d4r, d4i, d5r, d5i, d6r, d6i, d7r, d7i;
	vlf e0r, e0i, e2r, e2i, e3r, e3i, e4r, e4i;

	VLF_LD2(a0r, a0i, in + 0*VLF_WIDTH);
	VLF_LD2(a1r, a1i, in + 2*VLF_WIDTH);
	VLF_LD2(a2r, a2i, in + 4*VLF_WIDTH);
	VLF_LD2(a3r, a3i, in + 6*VLF_WIDTH);
	VLF_LD2(a4r, a4i, in + 8*VLF_WIDTH);
	VLF_LD2(a5r, a5i, in + 10*VLF_WIDTH);
	VLF_LD2(a6r, a6i, in + 12*VLF_WIDTH);
	VLF_LD2(a7r, a7i, in + 14*VLF_WIDTH);

	b0r = vlf_add(a0r, a4r);
	b4r = vlf_sub(a0r, a4r);
	b0i = vlf_add(a0i, a4i);
	b4i = vlf_sub(a0i, a4i);

	b1r = vlf_add(a1r, a5r);
	e0r = vlf_sub(a1r, a5r);
	b1i = vlf_add(a1i, a5i);
	e0i = vlf_sub(a5i, a1i);

	b2r = vlf_add(a2r, a6r);
	b6i = vlf_sub(a6r, a2r);
	b2i = vlf_add(a2i, a6i);
	b6r = vlf_sub(a2i, a6i);

	b3r = vlf_add(a3r, a7r);
	e2r = vlf_sub(a3r, a7r);
	b3i = vlf_add(a3i, a7i);
	e2i = vlf_sub(a3i, a7i);

	c0r = vlf_add(b0r, b2r);
	c2r = vlf_sub(b0r, b2r);
	c0i = vlf_add(b0i, b2i);
	c2i = vlf_sub(b0i, b2i);

	c1r = vlf_add(b1r, b3r);
	c3r = vlf_sub(b1r, b3r);
	c1i = vlf_add(b1i, b3i);
	c3i = vlf_sub(b1i, b3i);

	d0r = vlf_add(c0r, c1r);
	d4r = vlf_sub(c0r, c1r);
	d0i = vlf_add(c0i, c1i);
	d4i = vlf_sub(c0i, c1i);

	VLF_ST2(out + 0*outoffset, d0r, d0i);

	d2r = vlf_add(c2r, c3i);
	d6r = vlf_sub(c2r, c3i);
	d2i = vlf_sub(c2i, c3r);
	d6i = vlf_add(c2i, c3r);

	e3r = vlf_sub(e0i, e0r);
	e3i = vlf_add(e0r, e0i);
	e4r = vlf_sub(e2r, e2i);
	e4i = vlf_add(e2r, e2i);

	b5r = vlf_mul(e3r, vec_root_half);
	b5i = vlf_mul(e3i, vec_root_half);
	b7r = vlf_mul(e4r, vec_root_half);
	b7i = vlf_mul(e4i, vec_root_half);

	c4r = vlf_add(b4r, b6r);
	c6r = vlf_sub(b4r, b6r);
	c4i = vlf_add(b4i, b6i);
	c6i = vlf_sub(b4i, b6i);

	c5r = vlf_add(b5r, b7r);
	c7r = vlf_sub(b5r, b7r);
	c5i = vlf_add(b5i, b7i);
	c7i = vlf_sub(b5i, b7i);

	d1r = vlf_add(c4r, c5r);
	d5r = vlf_sub(c4r, c5r);
	d1i = vlf_add(c4i, c5i);
	d5i = vlf_sub(c4i, c5i);

	d3r = vlf_add(c6r, c7i);
	d7r = vlf_sub(c6r, c7i);
	d3i = vlf_sub(c6i, c7r);
	d7i = vlf_add(c6i, c7r);

	VLF_ST2(out + 1*outoffset, d1r, d1i);
	VLF_ST2(out + 2*outoffset, d2r, d2i);
	VLF_ST2(out + 3*outoffset, d3r, d3i);
	VLF_ST2(out + 4*outoffset, d4r, d4i);
	VLF_ST2(out + 5*outoffset, d5r, d5i);
	VLF_ST2(out + 6*outoffset, d6r, d6i);
	VLF_ST2(out + 7*outoffset, d7r, d7i);
}

static COP_ATTR_ALWAYSINLINE void fftset_vec_dif_fft16_offset_o(const float *in, float *out, unsigned outoffset)
{
	const vlf VC_C4 = vlf_broadcast(C_C4);
	const vlf VC_C8 = vlf_broadcast(C_C8);
	const vlf VC_S8 = vlf_broadcast(C_S8);

	float VEC_ALIGN_BEST stack[VLF_WIDTH*32];

	vlf a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i;
	vlf b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i;
	vlf c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i;
	vlf d0r, d0i, d1r, d1i, d2r, d2i, d3r, d3i;
	vlf y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
	vlf z0r, z0i, z1r, z1i, z2r, z2i, z3r, z3i;
	vlf e1r, e1i, e2r, e2i, e3r, e3i;

	VLF_LD2(a0r, a0i, in + 0*VLF_WIDTH);
	VLF_LD2(b0r, b0i, in + 8*VLF_WIDTH);
	VLF_LD2(c0r, c0i, in + 16*VLF_WIDTH);
	VLF_LD2(d0r, d0i, in + 24*VLF_WIDTH);

	/* fft 4 */
	y0r  = vlf_add(a0r, c0r);
	y0i  = vlf_add(a0i, c0i);
	y2r  = vlf_sub(a0r, c0r);
	y2i  = vlf_sub(a0i, c0i);
	y1r  = vlf_add(b0r, d0r);
	y1i  = vlf_add(b0i, d0i);
	y3r  = vlf_sub(b0r, d0r);
	y3i  = vlf_sub(b0i, d0i);
	z0r  = vlf_add(y0r, y1r);
	z0i  = vlf_add(y0i, y1i);
	z2r  = vlf_sub(y0r, y1r);
	z2i  = vlf_sub(y0i, y1i);
	z1r  = vlf_add(y2r, y3i);
	z1i  = vlf_sub(y2i, y3r);
	z3r  = vlf_sub(y2r, y3i);
	z3i  = vlf_add(y2i, y3r);

	VLF_ST2(stack + 0*VLF_WIDTH,  z0r, z0i);
	VLF_ST2(stack + 8*VLF_WIDTH,  z1r, z1i);
	VLF_ST2(stack + 16*VLF_WIDTH, z2r, z2i);
	VLF_ST2(stack + 24*VLF_WIDTH, z3r, z3i);
	VLF_LD2(a1r, a1i, in + 2*VLF_WIDTH);
	VLF_LD2(b1r, b1i, in + 10*VLF_WIDTH);
	VLF_LD2(c1r, c1i, in + 18*VLF_WIDTH);
	VLF_LD2(d1r, d1i, in + 26*VLF_WIDTH);

	/* fft 4 */
	y0r  = vlf_add(a1r, c1r);
	y2r  = vlf_sub(a1r, c1r);
	y0i  = vlf_add(a1i, c1i);
	y2i  = vlf_sub(a1i, c1i);
	y1r  = vlf_add(b1r, d1r);
	y3r  = vlf_sub(b1r, d1r);
	y1i  = vlf_add(b1i, d1i);
	y3i  = vlf_sub(b1i, d1i);
	z0r  = vlf_add(y0r, y1r);
	z2r  = vlf_sub(y0r, y1r);
	z0i  = vlf_add(y0i, y1i);
	z2i  = vlf_sub(y0i, y1i);
	z1r  = vlf_add(y2r, y3i);
	z3r  = vlf_sub(y2r, y3i);
	z1i  = vlf_sub(y2i, y3r);
	z3i  = vlf_add(y2i, y3r);

	/* twiddle */
	e1r  = vlf_mul(z1r, VC_C8);
	e1i  = vlf_mul(z1i, VC_C8);
	z1r  = vlf_mul(z1r, VC_S8);
	z1i  = vlf_mul(z1i, VC_S8);
	e1r  = vlf_add(e1r, z1i);
	e1i  = vlf_sub(e1i, z1r);
	z2r  = vlf_mul(z2r, VC_C4);
	z2i  = vlf_mul(z2i, VC_C4);
	e2r  = vlf_add(z2r, z2i);
	e2i  = vlf_sub(z2i, z2r);
	e3r  = vlf_mul(z3r, VC_S8);
	e3i  = vlf_mul(z3i, VC_S8);
	z3i  = vlf_mul(z3i, VC_C8);
	z3r  = vlf_mul(z3r, VC_C8);
	e3r  = vlf_add(e3r, z3i);
	e3i  = vlf_sub(e3i, z3r);

	VLF_ST2(stack + 2*VLF_WIDTH,  z0r, z0i);
	VLF_ST2(stack + 10*VLF_WIDTH, e1r, e1i);
	VLF_ST2(stack + 18*VLF_WIDTH, e2r, e2i);
	VLF_ST2(stack + 26*VLF_WIDTH, e3r, e3i);
	VLF_LD2(a2r, a2i, in + 4*VLF_WIDTH);
	VLF_LD2(b2r, b2i, in + 12*VLF_WIDTH);
	VLF_LD2(c2r, c2i, in + 20*VLF_WIDTH);
	VLF_LD2(d2r, d2i, in + 28*VLF_WIDTH);

	/* fft 4 */
	y0r  = vlf_add(a2r, c2r);
	y2r  = vlf_sub(a2r, c2r);
	y0i  = vlf_add(a2i, c2i);
	y2i  = vlf_sub(a2i, c2i);
	y1r  = vlf_add(b2r, d2r);
	y3r  = vlf_sub(b2r, d2r);
	y1i  = vlf_add(b2i, d2i);
	y3i  = vlf_sub(b2i, d2i);
	z0r  = vlf_add(y1r, y0r);
	z2r  = vlf_sub(y1r, y0r);
	z0i  = vlf_add(y0i, y1i);
	z2i  = vlf_sub(y0i, y1i);
	z1r  = vlf_add(y3i, y2r);
	z3r  = vlf_sub(y3i, y2r);
	z1i  = vlf_sub(y2i, y3r);
	z3i  = vlf_add(y2i, y3r);

	/* twiddle */
	e1r  = vlf_add(z1i, z1r);
	e1i  = vlf_sub(z1i, z1r);
	e3r  = vlf_add(z3r, z3i);
	e3i  = vlf_sub(z3r, z3i);
	e1r  = vlf_mul(e1r, VC_C4);
	e1i  = vlf_mul(e1i, VC_C4);
	e3r  = vlf_mul(e3r, VC_C4);
	e3i  = vlf_mul(e3i, VC_C4);

	VLF_ST2(stack + 4*VLF_WIDTH,  z0r, z0i);
	VLF_ST2(stack + 12*VLF_WIDTH, e1r, e1i);
	VLF_ST2(stack + 20*VLF_WIDTH, z2i, z2r);
	VLF_ST2(stack + 28*VLF_WIDTH, e3r, e3i);
	VLF_LD2(a3r, a3i, in + 6*VLF_WIDTH);
	VLF_LD2(b3r, b3i, in + 14*VLF_WIDTH);
	VLF_LD2(c3r, c3i, in + 22*VLF_WIDTH);
	VLF_LD2(d3r, d3i, in + 30*VLF_WIDTH);

	/* fft 4 */
	y0r  = vlf_add(a3r, c3r);
	y2r  = vlf_sub(a3r, c3r);
	y0i  = vlf_add(a3i, c3i);
	y2i  = vlf_sub(a3i, c3i);
	y1r  = vlf_add(b3r, d3r);
	y3r  = vlf_sub(d3r, b3r);
	y1i  = vlf_add(b3i, d3i);
	y3i  = vlf_sub(b3i, d3i);
	z0r  = vlf_add(y1r, y0r);
	z2r  = vlf_sub(y1r, y0r);
	z0i  = vlf_add(y0i, y1i);
	z2i  = vlf_sub(y0i, y1i);
	z1r  = vlf_add(y2r, y3i);
	z3r  = vlf_sub(y2r, y3i);
	z1i  = vlf_add(y3r, y2i);
	z3i  = vlf_sub(y3r, y2i);

	/* twiddle */
	e1r  = vlf_mul(z1r, VC_S8);
	e1i  = vlf_mul(z1i, VC_S8);
	z1i  = vlf_mul(z1i, VC_C8);
	z1r  = vlf_mul(z1r, VC_C8);
	e1r  = vlf_add(e1r, z1i);
	e1i  = vlf_sub(e1i, z1r);
	e2r  = vlf_add(z2r, z2i);
	e2i  = vlf_sub(z2r, z2i);
	e2r  = vlf_mul(e2r, VC_C4);
	e2i  = vlf_mul(e2i, VC_C4);
	e3r  = vlf_mul(z3i, VC_S8);
	e3i  = vlf_mul(z3r, VC_S8);
	z3r  = vlf_mul(z3r, VC_C8);
	z3i  = vlf_mul(z3i, VC_C8);
	e3r  = vlf_sub(e3r, z3r);
	e3i  = vlf_add(e3i, z3i);

	VLF_ST2(stack + 6*VLF_WIDTH,  z0r, z0i);
	fftset_vec_dif_fft4_offset_o(stack + 0*VLF_WIDTH,  out + 0*outoffset, 4*outoffset);

	VLF_ST2(stack + 14*VLF_WIDTH, e1r, e1i);
	fftset_vec_dif_fft4_offset_o(stack + 8*VLF_WIDTH,  out + 1*outoffset, 4*outoffset);

	VLF_ST2(stack + 22*VLF_WIDTH, e2r, e2i);
	fftset_vec_dif_fft4_offset_o(stack + 16*VLF_WIDTH, out + 2*outoffset, 4*outoffset);

	VLF_ST2(stack + 30*VLF_WIDTH, e3r, e3i);
	fftset_vec_dif_fft4_offset_o(stack + 24*VLF_WIDTH, out + 3*outoffset, 4*outoffset);
}

#define BUILD_INNER_PASSES(n_) \
static void fc_v4_r ## n_ ## _inner(float *work_buf, unsigned nfft, unsigned lfft, const float *twid) \
{ \
	assert(lfft == 1); \
	do { \
		fftset_vec_dif_fft ## n_ ## _offset_o(work_buf, work_buf, 2 * VLF_WIDTH); \
		work_buf += 2 * (n_) * VLF_WIDTH; \
	} while (--nfft); \
} \
static void fc_v4_stock_r ## n_ ## _inner(const float *in, float *out, const float *twid, unsigned ncol, unsigned nrow_div_radix) \
{ \
	const unsigned ioffset = (2*VLF_WIDTH)*nrow_div_radix; \
	assert(ncol == 1); \
	do { \
		fftset_vec_dif_fft ## n_ ## _offset_o(in, out, ioffset); \
		out += 2 * VLF_WIDTH; \
		in  += 2 * (n_) * VLF_WIDTH; \
	} while (--nrow_div_radix); \
}

#define BUILD_STANDARD_PASSES(n_, ntwid_) \
static void fc_v4_dif_r ## n_(float *work_buf, unsigned nfft, unsigned lfft, const float *twid) \
{ \
	unsigned rinc = lfft * (2 * VLF_WIDTH); \
	do { \
		unsigned j; \
		const float *tp = twid; \
		for (j = 0; j < lfft; j++, work_buf += 2 * VLF_WIDTH, tp += (ntwid_)) { \
			fftset_vec_dif_fft ## n_ ## _offset_io(work_buf, work_buf, tp, rinc, rinc); \
		} \
		work_buf += ((n_) - 1)*rinc; \
	} while (--nfft); \
} \
static void fc_v4_dit_r ## n_(float *work_buf, unsigned nfft, unsigned lfft, const float *twid) \
{ \
	unsigned rinc = lfft * (2 * VLF_WIDTH); \
	do { \
		unsigned j; \
		const float *tp = twid; \
		for (j = 0; j < lfft; j++, work_buf += 2 * VLF_WIDTH, tp += (ntwid_)) { \
			fftset_vec_dit_fft ## n_ ## _offset_io(work_buf, work_buf, tp, rinc, rinc); \
		} \
		work_buf += ((n_) - 1)*rinc; \
	} while (--nfft); \
} \
static void fc_v4_stock_r ## n_(const float *in, float *out, const float *twid, unsigned ncol, unsigned nrow_div_radix) \
{ \
	const unsigned ooffset = (2*VLF_WIDTH)*ncol; \
	const unsigned ioffset = ooffset*nrow_div_radix; \
	do { \
		const float *in0 = in; \
		const float *tp  = twid; \
		unsigned     j   = ncol; \
		do { \
			fftset_vec_dif_fft ## n_ ## _offset_io(in0, out, tp, ooffset, ioffset); \
			tp   += (ntwid_); \
			out  += (2*VLF_WIDTH); \
			in0  += (2*VLF_WIDTH); \
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
	} else if (   length % 4 == 0
			  &&  length / 4 != 8
			  &&  length / 4 != 4
			  ) {
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
