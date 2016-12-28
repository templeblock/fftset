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
#include <stdlib.h>

static const float C_2C5 = 0.309016994374948f; /* 2 * PI / 5 */
static const float C_2S5 = 0.951056516295154f; /* 2 * PI / 5 */

static const float C_C3  = 0.500000000000000f; /* PI / 3 */
static const float C_S3  = 0.866025403784440f; /* PI / 3 */

static const float C_C4  = 0.707106781186548f; /* PI / 4 */

static const float C_C5  = 0.809016994374947f; /* PI / 5 */
static const float C_S5  = 0.587785252292473f; /* PI / 5 */

static const float C_C8  = 0.923879532511288f; /* PI / 8 */
static const float C_S8  = 0.382683432365086f; /* PI / 8 */

#define BUILD_INNER_PASSES(vtyp_, ctyp_, vwidth_, n_) \
static void fftset_ ## vtyp_ ## _r ## n_ ## _inner(ctyp_ *work_buf, unsigned nfft, unsigned lfft, const ctyp_ *twid) \
{ \
	assert(lfft == 1); \
	do { \
		vtyp_ ## _dif_fft ## n_ ## _offset_o(work_buf, work_buf, 2 * vwidth_); \
		work_buf += 2 * (n_) * vwidth_; \
	} while (--nfft); \
} \
static void fftset_ ## vtyp_ ## _r ## n_ ## _inner_stock(const ctyp_ *in, ctyp_ *out, const ctyp_ *twid, unsigned ncol, unsigned nrow_div_radix) \
{ \
	const unsigned ioffset = (2*vwidth_)*nrow_div_radix; \
	assert(ncol == 1); \
	do { \
		vtyp_ ## _dif_fft ## n_ ## _offset_o(in, out, ioffset); \
		out += 2 * vwidth_; \
		in  += 2 * (n_) * vwidth_; \
	} while (--nrow_div_radix); \
}

#define BUILD_STANDARD_PASSES(vtyp_, ctyp_, vwidth_, n_, ntwid_) \
static void fftset_ ## vtyp_ ## _r ## n_ ## _dif(ctyp_ *work_buf, unsigned nfft, unsigned lfft, const ctyp_ *twid) \
{ \
	unsigned rinc = lfft * (2 * vwidth_); \
	do { \
		unsigned j; \
		const ctyp_ *tp = twid; \
		for (j = 0; j < lfft; j++, work_buf += 2 * vwidth_, tp += (ntwid_)) { \
			vtyp_ ## _dif_fft ## n_ ## _offset_io(work_buf, work_buf, tp, rinc, rinc); \
		} \
		work_buf += ((n_) - 1)*rinc; \
	} while (--nfft); \
} \
static void fftset_ ## vtyp_ ## _r ## n_ ## _dit(ctyp_ *work_buf, unsigned nfft, unsigned lfft, const ctyp_ *twid) \
{ \
	unsigned rinc = lfft * (2 * vwidth_); \
	do { \
		unsigned j; \
		const ctyp_ *tp = twid; \
		for (j = 0; j < lfft; j++, work_buf += 2 * vwidth_, tp += (ntwid_)) { \
			vtyp_ ## _dit_fft ## n_ ## _offset_io(work_buf, work_buf, tp, rinc, rinc); \
		} \
		work_buf += ((n_) - 1)*rinc; \
	} while (--nfft); \
} \
static void fftset_ ## vtyp_ ## _r ## n_ ## _stock(const ctyp_ *in, ctyp_ *out, const ctyp_ *twid, unsigned ncol, unsigned nrow_div_radix) \
{ \
	const unsigned ooffset = (2*vwidth_)*ncol; \
	const unsigned ioffset = ooffset*nrow_div_radix; \
	do { \
		const ctyp_ *in0 = in; \
		const ctyp_ *tp  = twid; \
		unsigned     j   = ncol; \
		do { \
			vtyp_ ## _dif_fft ## n_ ## _offset_io(in0, out, tp, ooffset, ioffset); \
			tp   += (ntwid_); \
			out  += (2*vwidth_); \
			in0  += (2*vwidth_); \
		} while (--j); \
		in = in + (n_)*ooffset; \
	} while (--nrow_div_radix); \
} \
BUILD_INNER_PASSES(vtyp_, ctyp_, vwidth_, n_)

#define VECRADIX2PASSES(vtyp_, vtyp_mac_, ctyp_, vwidth_) \
static COP_ATTR_ALWAYSINLINE void vtyp_ ## _dif_fft2_offset_io(const ctyp_ *in, ctyp_ *out, const ctyp_ *twid, unsigned in_stride, unsigned out_stride) \
{ \
	vtyp_ nre   = vtyp_ ## _ld(in + 0*vwidth_); \
	vtyp_ nim   = vtyp_ ## _ld(in + 1*vwidth_); \
	vtyp_ fre   = vtyp_ ## _ld(in + in_stride + 0*vwidth_); \
	vtyp_ fim   = vtyp_ ## _ld(in + in_stride + 1*vwidth_); \
	vtyp_ tre   = vtyp_ ## _broadcast(twid[0]); \
	vtyp_ tim   = vtyp_ ## _broadcast(twid[1]); \
	vtyp_ onre  = vtyp_ ## _add(nre, fre); \
	vtyp_ onim  = vtyp_ ## _add(nim, fim); \
	vtyp_ ptre  = vtyp_ ## _sub(nre, fre); \
	vtyp_ ptim  = vtyp_ ## _sub(nim, fim); \
	vtyp_ ofrea = vtyp_ ## _mul(ptre, tre); \
	vtyp_ ofreb = vtyp_ ## _mul(ptim, tim); \
	vtyp_ ofima = vtyp_ ## _mul(ptre, tim); \
	vtyp_ ofimb = vtyp_ ## _mul(ptim, tre); \
	vtyp_ ofre  = vtyp_ ## _sub(ofrea, ofreb); \
	vtyp_ ofim  = vtyp_ ## _add(ofima, ofimb); \
	vtyp_ ## _st(out + 0*vwidth_, onre); \
	vtyp_ ## _st(out + 1*vwidth_, onim); \
	vtyp_ ## _st(out + out_stride + 0*vwidth_, ofre); \
	vtyp_ ## _st(out + out_stride + 1*vwidth_, ofim); \
} \
static COP_ATTR_ALWAYSINLINE void vtyp_ ## _dif_fft2_offset_o(const ctyp_ *in, ctyp_ *out, unsigned out_stride) \
{ \
	vtyp_ nre, nim, fre, fim; \
	vtyp_ onre, onim, ofre, ofim; \
	vtyp_mac_ ## _LD2(nre, nim, in + 0*vwidth_); \
	vtyp_mac_ ## _LD2(fre, fim, in + 2*vwidth_); \
	onre = vtyp_ ## _add(nre, fre); \
	onim = vtyp_ ## _add(nim, fim); \
	ofre = vtyp_ ## _sub(nre, fre); \
	ofim = vtyp_ ## _sub(nim, fim); \
	vtyp_mac_ ## _ST2(out + 0, onre, onim); \
	vtyp_mac_ ## _ST2(out + out_stride, ofre, ofim); \
} \
static COP_ATTR_ALWAYSINLINE void vtyp_ ## _dit_fft2_offset_io(const ctyp_ *in, ctyp_ *out, const ctyp_ *twid, unsigned in_stride, unsigned out_stride) \
{ \
	vtyp_ nre  = vtyp_ ## _ld(in + 0*vwidth_); \
	vtyp_ nim  = vtyp_ ## _ld(in + 1*vwidth_); \
	vtyp_ ptre = vtyp_ ## _ld(in + in_stride + 0*vwidth_); \
	vtyp_ ptim = vtyp_ ## _ld(in + in_stride + 1*vwidth_); \
	vtyp_ tre  = vtyp_ ## _broadcast(twid[0]); \
	vtyp_ tim  = vtyp_ ## _broadcast(twid[1]); \
	vtyp_ frea = vtyp_ ## _mul(ptre, tre); \
	vtyp_ freb = vtyp_ ## _mul(ptim, tim); \
	vtyp_ fima = vtyp_ ## _mul(ptre, tim); \
	vtyp_ fimb = vtyp_ ## _mul(ptim, tre); \
	vtyp_ fre  = vtyp_ ## _sub(frea, freb); \
	vtyp_ fim  = vtyp_ ## _add(fima, fimb); \
	vtyp_ onre = vtyp_ ## _add(nre, fre); \
	vtyp_ onim = vtyp_ ## _add(nim, fim); \
	vtyp_ ofre = vtyp_ ## _sub(nre, fre); \
	vtyp_ ofim = vtyp_ ## _sub(nim, fim); \
	vtyp_ ## _st(out + 0*vwidth_, onre); \
	vtyp_ ## _st(out + 1*vwidth_, onim); \
	vtyp_ ## _st(out + out_stride + 0*vwidth_, ofre); \
	vtyp_ ## _st(out + out_stride + 1*vwidth_, ofim); \
} \
BUILD_STANDARD_PASSES(vtyp_, ctyp_, vwidth_, 2, 2)

#define VECRADIX3PASSES(vtyp_, vtyp_mac_, ctyp_, vwidth_) \
static COP_ATTR_ALWAYSINLINE void vtyp_ ## _dif_fft3_offset_io(const ctyp_ *in, ctyp_ *out, const ctyp_ *twid, unsigned in_stride, unsigned out_stride) \
{ \
	vtyp_ r0, i0, r1, i1, r2, i2; \
	vtyp_ or0, oi0, or1, oi1, or2, oi2; \
	vtyp_ ar1, ai1, ar2, ai2; \
	vtyp_ cr1, ci1, cr2, ci2; \
	vtyp_ dr1, di1, dr2, di2; \
	vtyp_ er1, ei1, er2, ei2; \
	vtyp_ tr1, ti1, tr2, ti2; \
	vtyp_ tr3, ti3, tr4, ti4; \
	vtyp_ tr5, ti5; \
	const vtyp_ coef0 = vtyp_ ## _broadcast(C_C3); \
	const vtyp_ coef1 = vtyp_ ## _broadcast(C_S3); \
	vtyp_mac_ ## _LD2(r0, i0, in + 0*in_stride); \
	vtyp_mac_ ## _LD2(r1, i1, in + 1*in_stride); \
	vtyp_mac_ ## _LD2(r2, i2, in + 2*in_stride); \
	tr1 = vtyp_ ## _add(r2, r1); \
	ti1 = vtyp_ ## _add(i2, i1); \
	tr2 = vtyp_ ## _sub(i2, i1); \
	ti2 = vtyp_ ## _sub(r2, r1); \
	tr5 = vtyp_ ## _mul(tr1, coef0); \
	ti5 = vtyp_ ## _mul(ti1, coef0); \
	tr3 = vtyp_ ## _mul(tr2, coef1); \
	ti3 = vtyp_ ## _mul(ti2, coef1); \
	tr4 = vtyp_ ## _sub(r0, tr5); \
	ti4 = vtyp_ ## _sub(i0, ti5); \
	or0 = vtyp_ ## _add(r0, tr1); \
	oi0 = vtyp_ ## _add(i0, ti1); \
	ar1 = vtyp_ ## _sub(tr4, tr3); \
	ai1 = vtyp_ ## _add(ti4, ti3); \
	ar2 = vtyp_ ## _add(tr4, tr3); \
	ai2 = vtyp_ ## _sub(ti4, ti3); \
	cr1 = vtyp_ ## _broadcast(twid[0]); \
	ci1 = vtyp_ ## _broadcast(twid[1]); \
	cr2 = vtyp_ ## _broadcast(twid[2]); \
	ci2 = vtyp_ ## _broadcast(twid[3]); \
	dr1 = vtyp_ ## _mul(ar1, cr1); \
	er1 = vtyp_ ## _mul(ai1, ci1); \
	di1 = vtyp_ ## _mul(ar1, ci1); \
	ei1 = vtyp_ ## _mul(ai1, cr1); \
	dr2 = vtyp_ ## _mul(ar2, cr2); \
	er2 = vtyp_ ## _mul(ai2, ci2); \
	di2 = vtyp_ ## _mul(ar2, ci2); \
	ei2 = vtyp_ ## _mul(ai2, cr2); \
	or1 = vtyp_ ## _sub(dr1, er1); \
	oi1 = vtyp_ ## _add(di1, ei1); \
	or2 = vtyp_ ## _sub(dr2, er2); \
	oi2 = vtyp_ ## _add(di2, ei2); \
	vtyp_mac_ ## _ST2(out + 0*out_stride, or0, oi0); \
	vtyp_mac_ ## _ST2(out + 1*out_stride, or1, oi1); \
	vtyp_mac_ ## _ST2(out + 2*out_stride, or2, oi2); \
} \
static COP_ATTR_ALWAYSINLINE void vtyp_ ## _dif_fft3_offset_o(const ctyp_ *in, ctyp_ *out, unsigned out_stride) \
{ \
	vtyp_ r0, i0, r1, i1, r2, i2; \
	vtyp_ or0, oi0, or1, oi1, or2, oi2; \
	vtyp_ tr1, ti1, tr2, ti2; \
	vtyp_ tr3, ti3, tr4, ti4; \
	vtyp_ tr5, ti5; \
	const vtyp_ coef0 = vtyp_ ## _broadcast(C_C3); \
	const vtyp_ coef1 = vtyp_ ## _broadcast(C_S3); \
	vtyp_mac_ ## _LD2(r0, i0, in + 0*vwidth_); \
	vtyp_mac_ ## _LD2(r1, i1, in + 2*vwidth_); \
	vtyp_mac_ ## _LD2(r2, i2, in + 4*vwidth_); \
	tr1 = vtyp_ ## _add(r2, r1); \
	ti1 = vtyp_ ## _add(i2, i1); \
	tr2 = vtyp_ ## _sub(i2, i1); \
	ti2 = vtyp_ ## _sub(r2, r1); \
	tr5 = vtyp_ ## _mul(tr1, coef0); \
	ti5 = vtyp_ ## _mul(ti1, coef0); \
	tr3 = vtyp_ ## _mul(tr2, coef1); \
	ti3 = vtyp_ ## _mul(ti2, coef1); \
	tr4 = vtyp_ ## _sub(r0, tr5); \
	ti4 = vtyp_ ## _sub(i0, ti5); \
	or0 = vtyp_ ## _add(r0, tr1); \
	oi0 = vtyp_ ## _add(i0, ti1); \
	or1 = vtyp_ ## _sub(tr4, tr3); \
	oi1 = vtyp_ ## _add(ti4, ti3); \
	or2 = vtyp_ ## _add(tr4, tr3); \
	oi2 = vtyp_ ## _sub(ti4, ti3); \
	vtyp_mac_ ## _ST2(out + 0*out_stride, or0, oi0); \
	vtyp_mac_ ## _ST2(out + 1*out_stride, or1, oi1); \
	vtyp_mac_ ## _ST2(out + 2*out_stride, or2, oi2); \
} \
static COP_ATTR_ALWAYSINLINE void vtyp_ ## _dit_fft3_offset_io(const ctyp_ *in, ctyp_ *out, const ctyp_ *twid, unsigned in_stride, unsigned out_stride) \
{ \
	vtyp_ r0, i0, r1, i1, r2, i2; \
	vtyp_ or0, oi0, or1, oi1, or2, oi2; \
	vtyp_ ar1, ai1, ar2, ai2; \
	vtyp_ cr1, ci1, cr2, ci2; \
	vtyp_ dr1, di1, dr2, di2; \
	vtyp_ er1, ei1, er2, ei2; \
	vtyp_ tr1, ti1, tr2, ti2; \
	vtyp_ tr3, ti3, tr4, ti4; \
	vtyp_ tr5, ti5; \
	const vtyp_ coef0 = vtyp_ ## _broadcast(C_C3); \
	const vtyp_ coef1 = vtyp_ ## _broadcast(C_S3); \
	vtyp_mac_ ## _LD2(r0, i0, in + 0*in_stride); \
	vtyp_mac_ ## _LD2(r1, i1, in + 1*in_stride); \
	vtyp_mac_ ## _LD2(r2, i2, in + 2*in_stride); \
	cr1 = vtyp_ ## _broadcast(twid[0]); \
	ci1 = vtyp_ ## _broadcast(twid[1]); \
	cr2 = vtyp_ ## _broadcast(twid[2]); \
	ci2 = vtyp_ ## _broadcast(twid[3]); \
	dr1 = vtyp_ ## _mul(r1, cr1); \
	er1 = vtyp_ ## _mul(i1, ci1); \
	di1 = vtyp_ ## _mul(r1, ci1); \
	ei1 = vtyp_ ## _mul(i1, cr1); \
	dr2 = vtyp_ ## _mul(r2, cr2); \
	er2 = vtyp_ ## _mul(i2, ci2); \
	di2 = vtyp_ ## _mul(r2, ci2); \
	ei2 = vtyp_ ## _mul(i2, cr2); \
	ar1 = vtyp_ ## _sub(dr1, er1); \
	ai1 = vtyp_ ## _add(di1, ei1); \
	ar2 = vtyp_ ## _sub(dr2, er2); \
	ai2 = vtyp_ ## _add(di2, ei2); \
	tr1 = vtyp_ ## _add(ar2, ar1); \
	ti1 = vtyp_ ## _add(ai2, ai1); \
	tr2 = vtyp_ ## _sub(ai2, ai1); \
	ti2 = vtyp_ ## _sub(ar2, ar1); \
	tr5 = vtyp_ ## _mul(tr1, coef0); \
	ti5 = vtyp_ ## _mul(ti1, coef0); \
	tr3 = vtyp_ ## _mul(tr2, coef1); \
	ti3 = vtyp_ ## _mul(ti2, coef1); \
	tr4 = vtyp_ ## _sub(r0, tr5); \
	ti4 = vtyp_ ## _sub(i0, ti5); \
	or0 = vtyp_ ## _add(r0, tr1); \
	oi0 = vtyp_ ## _add(i0, ti1); \
	or1 = vtyp_ ## _sub(tr4, tr3); \
	oi1 = vtyp_ ## _add(ti4, ti3); \
	or2 = vtyp_ ## _add(tr4, tr3); \
	oi2 = vtyp_ ## _sub(ti4, ti3); \
	vtyp_mac_ ## _ST2(out + 0*out_stride, or0, oi0); \
	vtyp_mac_ ## _ST2(out + 1*out_stride, or1, oi1); \
	vtyp_mac_ ## _ST2(out + 2*out_stride, or2, oi2); \
} \
BUILD_STANDARD_PASSES(vtyp_, ctyp_, vwidth_, 3, 4)

#define VECRADIX4PASSES(vtyp_, vtyp_mac_, ctyp_, vwidth_) \
static COP_ATTR_ALWAYSINLINE void vtyp_ ## _dif_fft4_offset_io(const ctyp_ *in, ctyp_ *out, const ctyp_ *twid, unsigned in_stride, unsigned out_stride) \
{ \
	vtyp_ b0r   = vtyp_ ## _ld(in + 0*in_stride + 0*vwidth_); \
	vtyp_ b0i   = vtyp_ ## _ld(in + 0*in_stride + 1*vwidth_); \
	vtyp_ b1r   = vtyp_ ## _ld(in + 1*in_stride + 0*vwidth_); \
	vtyp_ b1i   = vtyp_ ## _ld(in + 1*in_stride + 1*vwidth_); \
	vtyp_ b2r   = vtyp_ ## _ld(in + 2*in_stride + 0*vwidth_); \
	vtyp_ b2i   = vtyp_ ## _ld(in + 2*in_stride + 1*vwidth_); \
	vtyp_ b3r   = vtyp_ ## _ld(in + 3*in_stride + 0*vwidth_); \
	vtyp_ b3i   = vtyp_ ## _ld(in + 3*in_stride + 1*vwidth_); \
	vtyp_ yr0   = vtyp_ ## _add(b0r, b2r); \
	vtyp_ yi0   = vtyp_ ## _add(b0i, b2i); \
	vtyp_ yr2   = vtyp_ ## _sub(b0r, b2r); \
	vtyp_ yi2   = vtyp_ ## _sub(b0i, b2i); \
	vtyp_ yr1   = vtyp_ ## _add(b1r, b3r); \
	vtyp_ yi1   = vtyp_ ## _add(b1i, b3i); \
	vtyp_ yr3   = vtyp_ ## _sub(b1r, b3r); \
	vtyp_ yi3   = vtyp_ ## _sub(b1i, b3i); \
	vtyp_ tr0   = vtyp_ ## _add(yr0, yr1); \
	vtyp_ ti0   = vtyp_ ## _add(yi0, yi1); \
	vtyp_ tr2   = vtyp_ ## _sub(yr0, yr1); \
	vtyp_ ti2   = vtyp_ ## _sub(yi0, yi1); \
	vtyp_ tr1   = vtyp_ ## _add(yr2, yi3); \
	vtyp_ ti1   = vtyp_ ## _sub(yi2, yr3); \
	vtyp_ tr3   = vtyp_ ## _sub(yr2, yi3); \
	vtyp_ ti3   = vtyp_ ## _add(yi2, yr3); \
	vtyp_ c1r   = vtyp_ ## _broadcast(twid[0]); \
	vtyp_ c1i   = vtyp_ ## _broadcast(twid[1]); \
	vtyp_ c2r   = vtyp_ ## _broadcast(twid[2]); \
	vtyp_ c2i   = vtyp_ ## _broadcast(twid[3]); \
	vtyp_ c3r   = vtyp_ ## _broadcast(twid[4]); \
	vtyp_ c3i   = vtyp_ ## _broadcast(twid[5]); \
	vtyp_ o1ra  = vtyp_ ## _mul(tr1, c1r); \
	vtyp_ o1rb  = vtyp_ ## _mul(ti1, c1i); \
	vtyp_ o1ia  = vtyp_ ## _mul(tr1, c1i); \
	vtyp_ o1ib  = vtyp_ ## _mul(ti1, c1r); \
	vtyp_ o2ra  = vtyp_ ## _mul(tr2, c2r); \
	vtyp_ o2rb  = vtyp_ ## _mul(ti2, c2i); \
	vtyp_ o2ia  = vtyp_ ## _mul(tr2, c2i); \
	vtyp_ o2ib  = vtyp_ ## _mul(ti2, c2r); \
	vtyp_ o3ra  = vtyp_ ## _mul(tr3, c3r); \
	vtyp_ o3rb  = vtyp_ ## _mul(ti3, c3i); \
	vtyp_ o3ia  = vtyp_ ## _mul(tr3, c3i); \
	vtyp_ o3ib  = vtyp_ ## _mul(ti3, c3r); \
	vtyp_ o1r   = vtyp_ ## _sub(o1ra, o1rb); \
	vtyp_ o1i   = vtyp_ ## _add(o1ia, o1ib); \
	vtyp_ o2r   = vtyp_ ## _sub(o2ra, o2rb); \
	vtyp_ o2i   = vtyp_ ## _add(o2ia, o2ib); \
	vtyp_ o3r   = vtyp_ ## _sub(o3ra, o3rb); \
	vtyp_ o3i   = vtyp_ ## _add(o3ia, o3ib); \
	vtyp_ ## _st(out + 0*out_stride + 0*vwidth_, tr0); \
	vtyp_ ## _st(out + 0*out_stride + 1*vwidth_, ti0); \
	vtyp_ ## _st(out + 1*out_stride + 0*vwidth_, o1r); \
	vtyp_ ## _st(out + 1*out_stride + 1*vwidth_, o1i); \
	vtyp_ ## _st(out + 2*out_stride + 0*vwidth_, o2r); \
	vtyp_ ## _st(out + 2*out_stride + 1*vwidth_, o2i); \
	vtyp_ ## _st(out + 3*out_stride + 0*vwidth_, o3r); \
	vtyp_ ## _st(out + 3*out_stride + 1*vwidth_, o3i); \
} \
static COP_ATTR_ALWAYSINLINE void vtyp_ ## _dif_fft4_offset_o(const ctyp_ *in, ctyp_ *out, unsigned outoffset) \
{ \
	vtyp_ b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i; \
	vtyp_ y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i; \
	vtyp_ z0r, z0i, z1r, z1i, z2r, z2i, z3r, z3i; \
	vtyp_mac_ ## _LD2(b0r, b0i, in + 0*vwidth_); \
	vtyp_mac_ ## _LD2(b1r, b1i, in + 2*vwidth_); \
	vtyp_mac_ ## _LD2(b2r, b2i, in + 4*vwidth_); \
	vtyp_mac_ ## _LD2(b3r, b3i, in + 6*vwidth_); \
	y0r  = vtyp_ ## _add(b0r, b2r); \
	y0i  = vtyp_ ## _add(b0i, b2i); \
	y2r  = vtyp_ ## _sub(b0r, b2r); \
	y2i  = vtyp_ ## _sub(b0i, b2i); \
	y1r  = vtyp_ ## _add(b1r, b3r); \
	y1i  = vtyp_ ## _add(b1i, b3i); \
	y3r  = vtyp_ ## _sub(b1r, b3r); \
	y3i  = vtyp_ ## _sub(b1i, b3i); \
	z0r  = vtyp_ ## _add(y0r, y1r); \
	z0i  = vtyp_ ## _add(y0i, y1i); \
	z2r  = vtyp_ ## _sub(y0r, y1r); \
	z2i  = vtyp_ ## _sub(y0i, y1i); \
	z1r  = vtyp_ ## _add(y2r, y3i); \
	z1i  = vtyp_ ## _sub(y2i, y3r); \
	z3r  = vtyp_ ## _sub(y2r, y3i); \
	z3i  = vtyp_ ## _add(y2i, y3r); \
	vtyp_mac_ ## _ST2(out + 0*outoffset, z0r, z0i); \
	vtyp_mac_ ## _ST2(out + 1*outoffset, z1r, z1i); \
	vtyp_mac_ ## _ST2(out + 2*outoffset, z2r, z2i); \
	vtyp_mac_ ## _ST2(out + 3*outoffset, z3r, z3i); \
} \
static COP_ATTR_ALWAYSINLINE void vtyp_ ## _dit_fft4_offset_io(const ctyp_ *in, ctyp_ *out, const ctyp_ *twid, unsigned in_stride, unsigned out_stride) \
{ \
	vtyp_ b0r  = vtyp_ ## _ld(in + 0*in_stride + 0*vwidth_); \
	vtyp_ b0i  = vtyp_ ## _ld(in + 0*in_stride + 1*vwidth_); \
	vtyp_ b1r  = vtyp_ ## _ld(in + 1*in_stride + 0*vwidth_); \
	vtyp_ b1i  = vtyp_ ## _ld(in + 1*in_stride + 1*vwidth_); \
	vtyp_ b2r  = vtyp_ ## _ld(in + 2*in_stride + 0*vwidth_); \
	vtyp_ b2i  = vtyp_ ## _ld(in + 2*in_stride + 1*vwidth_); \
	vtyp_ b3r  = vtyp_ ## _ld(in + 3*in_stride + 0*vwidth_); \
	vtyp_ b3i  = vtyp_ ## _ld(in + 3*in_stride + 1*vwidth_); \
	vtyp_ c1r  = vtyp_ ## _broadcast(twid[0]); \
	vtyp_ c1i  = vtyp_ ## _broadcast(twid[1]); \
	vtyp_ c2r  = vtyp_ ## _broadcast(twid[2]); \
	vtyp_ c2i  = vtyp_ ## _broadcast(twid[3]); \
	vtyp_ c3r  = vtyp_ ## _broadcast(twid[4]); \
	vtyp_ c3i  = vtyp_ ## _broadcast(twid[5]); \
	vtyp_ x1ra = vtyp_ ## _mul(b1r, c1r); \
	vtyp_ x1rb = vtyp_ ## _mul(b1i, c1i); \
	vtyp_ x1ia = vtyp_ ## _mul(b1r, c1i); \
	vtyp_ x1ib = vtyp_ ## _mul(b1i, c1r); \
	vtyp_ x2ra = vtyp_ ## _mul(b2r, c2r); \
	vtyp_ x2rb = vtyp_ ## _mul(b2i, c2i); \
	vtyp_ x2ia = vtyp_ ## _mul(b2r, c2i); \
	vtyp_ x2ib = vtyp_ ## _mul(b2i, c2r); \
	vtyp_ x3ra = vtyp_ ## _mul(b3r, c3r); \
	vtyp_ x3rb = vtyp_ ## _mul(b3i, c3i); \
	vtyp_ x3ia = vtyp_ ## _mul(b3r, c3i); \
	vtyp_ x3ib = vtyp_ ## _mul(b3i, c3r); \
	vtyp_ x1r  = vtyp_ ## _sub(x1ra, x1rb); \
	vtyp_ x1i  = vtyp_ ## _add(x1ia, x1ib); \
	vtyp_ x2r  = vtyp_ ## _sub(x2ra, x2rb); \
	vtyp_ x2i  = vtyp_ ## _add(x2ia, x2ib); \
	vtyp_ x3r  = vtyp_ ## _sub(x3ra, x3rb); \
	vtyp_ x3i  = vtyp_ ## _add(x3ia, x3ib); \
	vtyp_ yr0  = vtyp_ ## _add(b0r, x2r); \
	vtyp_ yi0  = vtyp_ ## _add(b0i, x2i); \
	vtyp_ yr2  = vtyp_ ## _sub(b0r, x2r); \
	vtyp_ yi2  = vtyp_ ## _sub(b0i, x2i); \
	vtyp_ yr1  = vtyp_ ## _add(x1r, x3r); \
	vtyp_ yi1  = vtyp_ ## _add(x1i, x3i); \
	vtyp_ yr3  = vtyp_ ## _sub(x1r, x3r); \
	vtyp_ yi3  = vtyp_ ## _sub(x1i, x3i); \
	vtyp_ o0r  = vtyp_ ## _add(yr0, yr1); \
	vtyp_ o0i  = vtyp_ ## _add(yi0, yi1); \
	vtyp_ o2r  = vtyp_ ## _sub(yr0, yr1); \
	vtyp_ o2i  = vtyp_ ## _sub(yi0, yi1); \
	vtyp_ o1r  = vtyp_ ## _add(yr2, yi3); \
	vtyp_ o1i  = vtyp_ ## _sub(yi2, yr3); \
	vtyp_ o3r  = vtyp_ ## _sub(yr2, yi3); \
	vtyp_ o3i  = vtyp_ ## _add(yi2, yr3); \
	vtyp_ ## _st(out + 0*out_stride + 0*vwidth_, o0r); \
	vtyp_ ## _st(out + 0*out_stride + 1*vwidth_, o0i); \
	vtyp_ ## _st(out + 1*out_stride + 0*vwidth_, o1r); \
	vtyp_ ## _st(out + 1*out_stride + 1*vwidth_, o1i); \
	vtyp_ ## _st(out + 2*out_stride + 0*vwidth_, o2r); \
	vtyp_ ## _st(out + 2*out_stride + 1*vwidth_, o2i); \
	vtyp_ ## _st(out + 3*out_stride + 0*vwidth_, o3r); \
	vtyp_ ## _st(out + 3*out_stride + 1*vwidth_, o3i); \
} \
BUILD_STANDARD_PASSES(vtyp_, ctyp_, vwidth_, 4, 6)

#define VECRADIX5PASSES(vtyp_, vtyp_mac_, ctyp_, vwidth_) \
static COP_ATTR_ALWAYSINLINE void vtyp_ ## _dif_fft5_offset_o(const ctyp_ *in, ctyp_ *out, unsigned outoffset) \
{ \
	const vtyp_ c0r = vtyp_ ## _broadcast(C_2C5); \
	const vtyp_ c0i = vtyp_ ## _broadcast(C_2S5); \
	const vtyp_ c1r = vtyp_ ## _broadcast(C_C5); \
	const vtyp_ c1i = vtyp_ ## _broadcast(C_S5); \
	vtyp_ r0, i0, r1, r2, r3, r4, i1, i2, i3, i4; \
	vtyp_ a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i; \
	vtyp_ b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i; \
	vtyp_ d0r, d0i, d1r, d1i, d2r, d2i, d3r, d3i; \
	vtyp_ e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i; \
	vtyp_ y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i; \
	vtyp_ z0r, z0i, z1r, z1i, z2r, z2i; \
	vtyp_mac_ ## _LD2(r0, i0, in + 0*vwidth_); \
	vtyp_mac_ ## _LD2(r1, i1, in + 2*vwidth_); \
	vtyp_mac_ ## _LD2(r2, i2, in + 4*vwidth_); \
	vtyp_mac_ ## _LD2(r3, i3, in + 6*vwidth_); \
	vtyp_mac_ ## _LD2(r4, i4, in + 8*vwidth_); \
	a0r = vtyp_ ## _add(r2, r3); \
	a2r = vtyp_ ## _sub(r2, r3); \
	a1r = vtyp_ ## _add(r1, r4); \
	a3r = vtyp_ ## _sub(r1, r4); \
	a0i = vtyp_ ## _add(i2, i3); \
	a2i = vtyp_ ## _sub(i2, i3); \
	a1i = vtyp_ ## _add(i1, i4); \
	a3i = vtyp_ ## _sub(i1, i4); \
	z1r = vtyp_ ## _add(a0r, a1r); \
	z1i = vtyp_ ## _add(a0i, a1i); \
	y0r = vtyp_ ## _add(r0, z1r); \
	y0i = vtyp_ ## _add(i0, z1i); \
	vtyp_mac_ ## _ST2(out, y0r, y0i); \
	d0r = vtyp_ ## _mul(a3r, c0i); \
	e0r = vtyp_ ## _mul(a2r, c1i); \
	d0i = vtyp_ ## _mul(a3r, c1i); \
	e0i = vtyp_ ## _mul(a2r, c0i); \
	b1i = vtyp_ ## _add(d0r, e0r); \
	b3i = vtyp_ ## _sub(e0i, d0i); \
	d1r = vtyp_ ## _mul(a1r, c0r); \
	e1r = vtyp_ ## _mul(a0r, c1r); \
	d1i = vtyp_ ## _mul(a1r, c1r); \
	e1i = vtyp_ ## _mul(a0r, c0r); \
	b0r = vtyp_ ## _sub(d1r, e1r); \
	b2r = vtyp_ ## _sub(e1i, d1i); \
	d2r = vtyp_ ## _mul(a3i, c0i); \
	e2r = vtyp_ ## _mul(a2i, c1i); \
	d2i = vtyp_ ## _mul(a3i, c1i); \
	e2i = vtyp_ ## _mul(a2i, c0i); \
	b1r = vtyp_ ## _add(d2r, e2r); \
	b3r = vtyp_ ## _sub(e2i, d2i); \
	d3r = vtyp_ ## _mul(a1i, c0r); \
	e3r = vtyp_ ## _mul(a0i, c1r); \
	d3i = vtyp_ ## _mul(a1i, c1r); \
	e3i = vtyp_ ## _mul(a0i, c0r); \
	b0i = vtyp_ ## _sub(d3r, e3r); \
	b2i = vtyp_ ## _sub(e3i, d3i); \
	z0r = vtyp_ ## _add(b0r, r0); \
	z2r = vtyp_ ## _add(b2r, r0); \
	z0i = vtyp_ ## _add(b0i, i0); \
	z2i = vtyp_ ## _add(b2i, i0); \
	y1r = vtyp_ ## _add(z0r, b1r); \
	y1i = vtyp_ ## _sub(z0i, b1i); \
	y4r = vtyp_ ## _sub(z0r, b1r); \
	y4i = vtyp_ ## _add(z0i, b1i); \
	y2r = vtyp_ ## _sub(z2r, b3r); \
	y2i = vtyp_ ## _add(z2i, b3i); \
	y3r = vtyp_ ## _add(z2r, b3r); \
	y3i = vtyp_ ## _sub(z2i, b3i); \
	vtyp_mac_ ## _ST2(out + 1*outoffset, y1r, y1i); \
	vtyp_mac_ ## _ST2(out + 2*outoffset, y2r, y2i); \
	vtyp_mac_ ## _ST2(out + 3*outoffset, y3r, y3i); \
	vtyp_mac_ ## _ST2(out + 4*outoffset, y4r, y4i); \
} \
BUILD_INNER_PASSES(vtyp_, ctyp_, vwidth_, 5)

#define VECRADIX6PASSES(vtyp_, vtyp_mac_, ctyp_, vwidth_) \
static COP_ATTR_ALWAYSINLINE void vtyp_ ## _dif_fft6_offset_o(const ctyp_ *in, ctyp_ *out, unsigned outoffset) \
{ \
	const vtyp_ c0r = vtyp_ ## _broadcast(C_C3); \
	const vtyp_ c0i = vtyp_ ## _broadcast(C_S3); \
	vtyp_ a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, a4r, a4i, a5r, a5i; \
	vtyp_ b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i, b4r, b4i, b5r, b5i; \
	vtyp_ c1r, c1i, c2r, c2i, c3r, c3i, c4r, c4i; \
	vtyp_ d1r, d1i, d2r, d2i, d3r, d3i, d4r, d4i; \
	vtyp_ e1r, e1i, e2r, e2i; \
	vtyp_ f0r, f0i, f1r, f1i, f2r, f2i, f3r, f3i, f4r, f4i, f5r, f5i; \
	vtyp_mac_ ## _LD2(a0r, a0i, in + 0*vwidth_); \
	vtyp_mac_ ## _LD2(a1r, a1i, in + 2*vwidth_); \
	vtyp_mac_ ## _LD2(a2r, a2i, in + 4*vwidth_); \
	vtyp_mac_ ## _LD2(a3r, a3i, in + 6*vwidth_); \
	vtyp_mac_ ## _LD2(a4r, a4i, in + 8*vwidth_); \
	vtyp_mac_ ## _LD2(a5r, a5i, in + 10*vwidth_); \
	b2r = vtyp_ ## _add(a1r, a5r); \
	b2i = vtyp_ ## _add(a1i, a5i); \
	b3r = vtyp_ ## _sub(a1r, a5r); \
	b3i = vtyp_ ## _sub(a1i, a5i); \
	b4r = vtyp_ ## _add(a2r, a4r); \
	b4i = vtyp_ ## _add(a2i, a4i); \
	b5r = vtyp_ ## _sub(a2r, a4r); \
	b5i = vtyp_ ## _sub(a2i, a4i); \
	c1r = vtyp_ ## _add(b2r, b4r); \
	c1i = vtyp_ ## _add(b2i, b4i); \
	c2r = vtyp_ ## _sub(b2r, b4r); \
	c2i = vtyp_ ## _sub(b2i, b4i); \
	c3r = vtyp_ ## _add(b3r, b5r); \
	c3i = vtyp_ ## _add(b3i, b5i); \
	c4r = vtyp_ ## _sub(b3r, b5r); \
	c4i = vtyp_ ## _sub(b3i, b5i); \
	d1r = vtyp_ ## _mul(c1r, c0r); \
	d1i = vtyp_ ## _mul(c1i, c0r); \
	d2r = vtyp_ ## _mul(c2r, c0r); \
	d2i = vtyp_ ## _mul(c2i, c0r); \
	d3r = vtyp_ ## _mul(c3r, c0i); \
	d3i = vtyp_ ## _mul(c3i, c0i); \
	d4r = vtyp_ ## _mul(c4r, c0i); \
	d4i = vtyp_ ## _mul(c4i, c0i); \
	b0r = vtyp_ ## _add(a0r, a3r); \
	b0i = vtyp_ ## _add(a0i, a3i); \
	b1r = vtyp_ ## _sub(a0r, a3r); \
	b1i = vtyp_ ## _sub(a0i, a3i); \
	f0r = vtyp_ ## _add(b0r, c1r); \
	f0i = vtyp_ ## _add(b0i, c1i); \
	e1r = vtyp_ ## _sub(b0r, d1r); \
	e1i = vtyp_ ## _sub(b0i, d1i); \
	f3r = vtyp_ ## _sub(b1r, c2r); \
	f3i = vtyp_ ## _sub(b1i, c2i); \
	e2r = vtyp_ ## _add(b1r, d2r); \
	e2i = vtyp_ ## _add(b1i, d2i); \
	f2r = vtyp_ ## _add(e1r, d4i); \
	f2i = vtyp_ ## _sub(e1i, d4r); \
	f4r = vtyp_ ## _sub(e1r, d4i); \
	f4i = vtyp_ ## _add(e1i, d4r); \
	f1r = vtyp_ ## _add(e2r, d3i); \
	f1i = vtyp_ ## _sub(e2i, d3r); \
	f5r = vtyp_ ## _sub(e2r, d3i); \
	f5i = vtyp_ ## _add(e2i, d3r); \
	vtyp_mac_ ## _ST2(out + 0*outoffset, f0r, f0i); \
	vtyp_mac_ ## _ST2(out + 1*outoffset, f1r, f1i); \
	vtyp_mac_ ## _ST2(out + 2*outoffset, f2r, f2i); \
	vtyp_mac_ ## _ST2(out + 3*outoffset, f3r, f3i); \
	vtyp_mac_ ## _ST2(out + 4*outoffset, f4r, f4i); \
	vtyp_mac_ ## _ST2(out + 5*outoffset, f5r, f5i); \
} \
BUILD_INNER_PASSES(vtyp_, ctyp_, vwidth_, 6)

#define VECRADIX8PASSES(vtyp_, vtyp_mac_, ctyp_, vwidth_) \
static COP_ATTR_ALWAYSINLINE void vtyp_ ## _dif_fft8_offset_o(const ctyp_ *in, ctyp_ *out, unsigned outoffset) \
{ \
	const vtyp_ vec_root_half = vtyp_ ## _broadcast(C_C4); \
	vtyp_ a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, a4r, a4i, a5r, a5i, a6r, a6i, a7r, a7i; \
	vtyp_ b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i, b4r, b4i, b5r, b5i, b6r, b6i, b7r, b7i; \
	vtyp_ c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i, c4r, c4i, c5r, c5i, c6r, c6i, c7r, c7i; \
	vtyp_ d0r, d0i, d1r, d1i, d2r, d2i, d3r, d3i, d4r, d4i, d5r, d5i, d6r, d6i, d7r, d7i; \
	vtyp_ e0r, e0i, e2r, e2i, e3r, e3i, e4r, e4i; \
	vtyp_mac_ ## _LD2(a0r, a0i, in + 0*vwidth_); \
	vtyp_mac_ ## _LD2(a1r, a1i, in + 2*vwidth_); \
	vtyp_mac_ ## _LD2(a2r, a2i, in + 4*vwidth_); \
	vtyp_mac_ ## _LD2(a3r, a3i, in + 6*vwidth_); \
	vtyp_mac_ ## _LD2(a4r, a4i, in + 8*vwidth_); \
	vtyp_mac_ ## _LD2(a5r, a5i, in + 10*vwidth_); \
	vtyp_mac_ ## _LD2(a6r, a6i, in + 12*vwidth_); \
	vtyp_mac_ ## _LD2(a7r, a7i, in + 14*vwidth_); \
	b0r = vtyp_ ## _add(a0r, a4r); \
	b4r = vtyp_ ## _sub(a0r, a4r); \
	b0i = vtyp_ ## _add(a0i, a4i); \
	b4i = vtyp_ ## _sub(a0i, a4i); \
	b1r = vtyp_ ## _add(a1r, a5r); \
	e0r = vtyp_ ## _sub(a1r, a5r); \
	b1i = vtyp_ ## _add(a1i, a5i); \
	e0i = vtyp_ ## _sub(a5i, a1i); \
	b2r = vtyp_ ## _add(a2r, a6r); \
	b6i = vtyp_ ## _sub(a6r, a2r); \
	b2i = vtyp_ ## _add(a2i, a6i); \
	b6r = vtyp_ ## _sub(a2i, a6i); \
	b3r = vtyp_ ## _add(a3r, a7r); \
	e2r = vtyp_ ## _sub(a3r, a7r); \
	b3i = vtyp_ ## _add(a3i, a7i); \
	e2i = vtyp_ ## _sub(a3i, a7i); \
	c0r = vtyp_ ## _add(b0r, b2r); \
	c2r = vtyp_ ## _sub(b0r, b2r); \
	c0i = vtyp_ ## _add(b0i, b2i); \
	c2i = vtyp_ ## _sub(b0i, b2i); \
	c1r = vtyp_ ## _add(b1r, b3r); \
	c3r = vtyp_ ## _sub(b1r, b3r); \
	c1i = vtyp_ ## _add(b1i, b3i); \
	c3i = vtyp_ ## _sub(b1i, b3i); \
	d0r = vtyp_ ## _add(c0r, c1r); \
	d4r = vtyp_ ## _sub(c0r, c1r); \
	d0i = vtyp_ ## _add(c0i, c1i); \
	d4i = vtyp_ ## _sub(c0i, c1i); \
	vtyp_mac_ ## _ST2(out + 0*outoffset, d0r, d0i); \
	d2r = vtyp_ ## _add(c2r, c3i); \
	d6r = vtyp_ ## _sub(c2r, c3i); \
	d2i = vtyp_ ## _sub(c2i, c3r); \
	d6i = vtyp_ ## _add(c2i, c3r); \
	e3r = vtyp_ ## _sub(e0i, e0r); \
	e3i = vtyp_ ## _add(e0r, e0i); \
	e4r = vtyp_ ## _sub(e2r, e2i); \
	e4i = vtyp_ ## _add(e2r, e2i); \
	b5r = vtyp_ ## _mul(e3r, vec_root_half); \
	b5i = vtyp_ ## _mul(e3i, vec_root_half); \
	b7r = vtyp_ ## _mul(e4r, vec_root_half); \
	b7i = vtyp_ ## _mul(e4i, vec_root_half); \
	c4r = vtyp_ ## _add(b4r, b6r); \
	c6r = vtyp_ ## _sub(b4r, b6r); \
	c4i = vtyp_ ## _add(b4i, b6i); \
	c6i = vtyp_ ## _sub(b4i, b6i); \
	c5r = vtyp_ ## _add(b5r, b7r); \
	c7r = vtyp_ ## _sub(b7r, b5r); \
	c5i = vtyp_ ## _add(b5i, b7i); \
	c7i = vtyp_ ## _sub(b7i, b5i); \
	d1r = vtyp_ ## _sub(c4r, c5r); \
	d5r = vtyp_ ## _add(c4r, c5r); \
	d1i = vtyp_ ## _sub(c4i, c5i); \
	d5i = vtyp_ ## _add(c4i, c5i); \
	d3r = vtyp_ ## _add(c6r, c7i); \
	d7r = vtyp_ ## _sub(c6r, c7i); \
	d3i = vtyp_ ## _sub(c6i, c7r); \
	d7i = vtyp_ ## _add(c6i, c7r); \
	vtyp_mac_ ## _ST2(out + 1*outoffset, d1r, d1i); \
	vtyp_mac_ ## _ST2(out + 2*outoffset, d2r, d2i); \
	vtyp_mac_ ## _ST2(out + 3*outoffset, d3r, d3i); \
	vtyp_mac_ ## _ST2(out + 4*outoffset, d4r, d4i); \
	vtyp_mac_ ## _ST2(out + 5*outoffset, d5r, d5i); \
	vtyp_mac_ ## _ST2(out + 6*outoffset, d6r, d6i); \
	vtyp_mac_ ## _ST2(out + 7*outoffset, d7r, d7i); \
} \
BUILD_INNER_PASSES(vtyp_, ctyp_, vwidth_, 8)

#define VECRADIX16PASSES(vtyp_, vtyp_mac_, ctyp_, vwidth_) \
static COP_ATTR_ALWAYSINLINE void vtyp_ ## _dif_fft16_offset_o(const ctyp_ *in, ctyp_ *out, unsigned outoffset) \
{ \
	const vtyp_ VC_C4 = vtyp_ ## _broadcast(C_C4); \
	const vtyp_ VC_C8 = vtyp_ ## _broadcast(C_C8); \
	const vtyp_ VC_S8 = vtyp_ ## _broadcast(C_S8); \
	ctyp_ VEC_ALIGN_BEST stack[vwidth_*32]; \
	vtyp_ a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i; \
	vtyp_ b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i; \
	vtyp_ c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i; \
	vtyp_ d0r, d0i, d1r, d1i, d2r, d2i, d3r, d3i; \
	vtyp_ y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i; \
	vtyp_ z0r, z0i, z1r, z1i, z2r, z2i, z3r, z3i; \
	vtyp_ e1r, e1i, e2r, e2i, e3r, e3i; \
	vtyp_mac_ ## _LD2(a0r, a0i, in + 0*vwidth_); \
	vtyp_mac_ ## _LD2(b0r, b0i, in + 8*vwidth_); \
	vtyp_mac_ ## _LD2(c0r, c0i, in + 16*vwidth_); \
	vtyp_mac_ ## _LD2(d0r, d0i, in + 24*vwidth_); \
	y0r  = vtyp_ ## _add(a0r, c0r); \
	y0i  = vtyp_ ## _add(a0i, c0i); \
	y2r  = vtyp_ ## _sub(a0r, c0r); \
	y2i  = vtyp_ ## _sub(a0i, c0i); \
	y1r  = vtyp_ ## _add(b0r, d0r); \
	y1i  = vtyp_ ## _add(b0i, d0i); \
	y3r  = vtyp_ ## _sub(b0r, d0r); \
	y3i  = vtyp_ ## _sub(b0i, d0i); \
	z0r  = vtyp_ ## _add(y0r, y1r); \
	z0i  = vtyp_ ## _add(y0i, y1i); \
	z2r  = vtyp_ ## _sub(y0r, y1r); \
	z2i  = vtyp_ ## _sub(y0i, y1i); \
	z1r  = vtyp_ ## _add(y2r, y3i); \
	z1i  = vtyp_ ## _sub(y2i, y3r); \
	z3r  = vtyp_ ## _sub(y2r, y3i); \
	z3i  = vtyp_ ## _add(y2i, y3r); \
	vtyp_mac_ ## _ST2(stack + 0*vwidth_,  z0r, z0i); \
	vtyp_mac_ ## _ST2(stack + 8*vwidth_,  z1r, z1i); \
	vtyp_mac_ ## _ST2(stack + 16*vwidth_, z2r, z2i); \
	vtyp_mac_ ## _ST2(stack + 24*vwidth_, z3r, z3i); \
	vtyp_mac_ ## _LD2(a1r, a1i, in + 2*vwidth_); \
	vtyp_mac_ ## _LD2(b1r, b1i, in + 10*vwidth_); \
	vtyp_mac_ ## _LD2(c1r, c1i, in + 18*vwidth_); \
	vtyp_mac_ ## _LD2(d1r, d1i, in + 26*vwidth_); \
	y0r  = vtyp_ ## _add(a1r, c1r); \
	y2r  = vtyp_ ## _sub(a1r, c1r); \
	y0i  = vtyp_ ## _add(a1i, c1i); \
	y2i  = vtyp_ ## _sub(a1i, c1i); \
	y1r  = vtyp_ ## _add(b1r, d1r); \
	y3r  = vtyp_ ## _sub(b1r, d1r); \
	y1i  = vtyp_ ## _add(b1i, d1i); \
	y3i  = vtyp_ ## _sub(b1i, d1i); \
	z0r  = vtyp_ ## _add(y0r, y1r); \
	z2r  = vtyp_ ## _sub(y0r, y1r); \
	z0i  = vtyp_ ## _add(y0i, y1i); \
	z2i  = vtyp_ ## _sub(y0i, y1i); \
	z1r  = vtyp_ ## _add(y2r, y3i); \
	z3r  = vtyp_ ## _sub(y2r, y3i); \
	z1i  = vtyp_ ## _sub(y2i, y3r); \
	z3i  = vtyp_ ## _add(y2i, y3r); \
	e1r  = vtyp_ ## _mul(z1r, VC_C8); \
	e1i  = vtyp_ ## _mul(z1i, VC_C8); \
	z1r  = vtyp_ ## _mul(z1r, VC_S8); \
	z1i  = vtyp_ ## _mul(z1i, VC_S8); \
	e1r  = vtyp_ ## _add(e1r, z1i); \
	e1i  = vtyp_ ## _sub(e1i, z1r); \
	z2r  = vtyp_ ## _mul(z2r, VC_C4); \
	z2i  = vtyp_ ## _mul(z2i, VC_C4); \
	e2r  = vtyp_ ## _add(z2r, z2i); \
	e2i  = vtyp_ ## _sub(z2i, z2r); \
	e3r  = vtyp_ ## _mul(z3r, VC_S8); \
	e3i  = vtyp_ ## _mul(z3i, VC_S8); \
	z3i  = vtyp_ ## _mul(z3i, VC_C8); \
	z3r  = vtyp_ ## _mul(z3r, VC_C8); \
	e3r  = vtyp_ ## _add(e3r, z3i); \
	e3i  = vtyp_ ## _sub(e3i, z3r); \
	vtyp_mac_ ## _ST2(stack + 2*vwidth_,  z0r, z0i); \
	vtyp_mac_ ## _ST2(stack + 10*vwidth_, e1r, e1i); \
	vtyp_mac_ ## _ST2(stack + 18*vwidth_, e2r, e2i); \
	vtyp_mac_ ## _ST2(stack + 26*vwidth_, e3r, e3i); \
	vtyp_mac_ ## _LD2(a2r, a2i, in + 4*vwidth_); \
	vtyp_mac_ ## _LD2(b2r, b2i, in + 12*vwidth_); \
	vtyp_mac_ ## _LD2(c2r, c2i, in + 20*vwidth_); \
	vtyp_mac_ ## _LD2(d2r, d2i, in + 28*vwidth_); \
	y0r  = vtyp_ ## _add(a2r, c2r); \
	y2r  = vtyp_ ## _sub(a2r, c2r); \
	y0i  = vtyp_ ## _add(a2i, c2i); \
	y2i  = vtyp_ ## _sub(a2i, c2i); \
	y1r  = vtyp_ ## _add(b2r, d2r); \
	y3r  = vtyp_ ## _sub(b2r, d2r); \
	y1i  = vtyp_ ## _add(b2i, d2i); \
	y3i  = vtyp_ ## _sub(b2i, d2i); \
	z0r  = vtyp_ ## _add(y1r, y0r); \
	z2r  = vtyp_ ## _sub(y1r, y0r); \
	z0i  = vtyp_ ## _add(y0i, y1i); \
	z2i  = vtyp_ ## _sub(y0i, y1i); \
	z1r  = vtyp_ ## _add(y3i, y2r); \
	z3r  = vtyp_ ## _sub(y3i, y2r); \
	z1i  = vtyp_ ## _sub(y2i, y3r); \
	z3i  = vtyp_ ## _add(y2i, y3r); \
	e1r  = vtyp_ ## _add(z1i, z1r); \
	e1i  = vtyp_ ## _sub(z1i, z1r); \
	e3r  = vtyp_ ## _add(z3r, z3i); \
	e3i  = vtyp_ ## _sub(z3r, z3i); \
	e1r  = vtyp_ ## _mul(e1r, VC_C4); \
	e1i  = vtyp_ ## _mul(e1i, VC_C4); \
	e3r  = vtyp_ ## _mul(e3r, VC_C4); \
	e3i  = vtyp_ ## _mul(e3i, VC_C4); \
	vtyp_mac_ ## _ST2(stack + 4*vwidth_,  z0r, z0i); \
	vtyp_mac_ ## _ST2(stack + 12*vwidth_, e1r, e1i); \
	vtyp_mac_ ## _ST2(stack + 20*vwidth_, z2i, z2r); \
	vtyp_mac_ ## _ST2(stack + 28*vwidth_, e3r, e3i); \
	vtyp_mac_ ## _LD2(a3r, a3i, in + 6*vwidth_); \
	vtyp_mac_ ## _LD2(b3r, b3i, in + 14*vwidth_); \
	vtyp_mac_ ## _LD2(c3r, c3i, in + 22*vwidth_); \
	vtyp_mac_ ## _LD2(d3r, d3i, in + 30*vwidth_); \
	y0r  = vtyp_ ## _add(a3r, c3r); \
	y2r  = vtyp_ ## _sub(a3r, c3r); \
	y0i  = vtyp_ ## _add(a3i, c3i); \
	y2i  = vtyp_ ## _sub(a3i, c3i); \
	y1r  = vtyp_ ## _add(b3r, d3r); \
	y3r  = vtyp_ ## _sub(d3r, b3r); \
	y1i  = vtyp_ ## _add(b3i, d3i); \
	y3i  = vtyp_ ## _sub(b3i, d3i); \
	z0r  = vtyp_ ## _add(y1r, y0r); \
	z2r  = vtyp_ ## _sub(y1r, y0r); \
	z0i  = vtyp_ ## _add(y0i, y1i); \
	z2i  = vtyp_ ## _sub(y0i, y1i); \
	z1r  = vtyp_ ## _add(y2r, y3i); \
	z3r  = vtyp_ ## _sub(y2r, y3i); \
	z1i  = vtyp_ ## _add(y3r, y2i); \
	z3i  = vtyp_ ## _sub(y3r, y2i); \
	e1r  = vtyp_ ## _mul(z1r, VC_S8); \
	e1i  = vtyp_ ## _mul(z1i, VC_S8); \
	z1i  = vtyp_ ## _mul(z1i, VC_C8); \
	z1r  = vtyp_ ## _mul(z1r, VC_C8); \
	e1r  = vtyp_ ## _add(e1r, z1i); \
	e1i  = vtyp_ ## _sub(e1i, z1r); \
	e2r  = vtyp_ ## _add(z2r, z2i); \
	e2i  = vtyp_ ## _sub(z2r, z2i); \
	e2r  = vtyp_ ## _mul(e2r, VC_C4); \
	e2i  = vtyp_ ## _mul(e2i, VC_C4); \
	e3r  = vtyp_ ## _mul(z3i, VC_S8); \
	e3i  = vtyp_ ## _mul(z3r, VC_S8); \
	z3r  = vtyp_ ## _mul(z3r, VC_C8); \
	z3i  = vtyp_ ## _mul(z3i, VC_C8); \
	e3r  = vtyp_ ## _sub(e3r, z3r); \
	e3i  = vtyp_ ## _add(e3i, z3i); \
	vtyp_mac_ ## _ST2(stack + 6*vwidth_,  z0r, z0i); \
	vtyp_ ## _dif_fft4_offset_o(stack + 0*vwidth_,  out + 0*outoffset, 4*outoffset); \
	vtyp_mac_ ## _ST2(stack + 14*vwidth_, e1r, e1i); \
	vtyp_ ## _dif_fft4_offset_o(stack + 8*vwidth_,  out + 1*outoffset, 4*outoffset); \
	vtyp_mac_ ## _ST2(stack + 22*vwidth_, e2r, e2i); \
	vtyp_ ## _dif_fft4_offset_o(stack + 16*vwidth_, out + 2*outoffset, 4*outoffset); \
	vtyp_mac_ ## _ST2(stack + 30*vwidth_, e3r, e3i); \
	vtyp_ ## _dif_fft4_offset_o(stack + 24*vwidth_, out + 3*outoffset, 4*outoffset); \
} \
BUILD_INNER_PASSES(vtyp_, ctyp_, vwidth_, 16)

#if 1
VECRADIX2PASSES(v1f, V1F, float, 1)
VECRADIX3PASSES(v1f, V1F, float, 1)
VECRADIX4PASSES(v1f, V1F, float, 1)
VECRADIX5PASSES(v1f, V1F, float, 1)
VECRADIX6PASSES(v1f, V1F, float, 1)
VECRADIX8PASSES(v1f, V1F, float, 1)
VECRADIX16PASSES(v1f, V1F, float, 1)
#endif

#if 0
VECRADIX2PASSES(v1d, V1D, double, 1)
VECRADIX3PASSES(v1d, V1D, double, 1)
VECRADIX4PASSES(v1d, V1D, double, 1)
VECRADIX5PASSES(v1d, V1D, double, 1)
VECRADIX6PASSES(v1d, V1D, double, 1)
VECRADIX8PASSES(v1d, V1D, double, 1)
VECRADIX16PASSES(v1d, V1D, double, 1)
#endif

#if V4F_EXISTS
VECRADIX2PASSES(v4f, V4F, float, 4)
VECRADIX3PASSES(v4f, V4F, float, 4)
VECRADIX4PASSES(v4f, V4F, float, 4)
VECRADIX5PASSES(v4f, V4F, float, 4)
VECRADIX6PASSES(v4f, V4F, float, 4)
VECRADIX8PASSES(v4f, V4F, float, 4)
VECRADIX16PASSES(v4f, V4F, float, 4)
#endif

#if V8F_EXISTS
VECRADIX2PASSES(v8f, V8F, float, 8)
VECRADIX3PASSES(v8f, V8F, float, 8)
VECRADIX4PASSES(v8f, V8F, float, 8)
VECRADIX5PASSES(v8f, V8F, float, 8)
VECRADIX6PASSES(v8f, V8F, float, 8)
VECRADIX8PASSES(v8f, V8F, float, 8)
VECRADIX16PASSES(v8f, V8F, float, 8)
#endif

struct float_pass_radix {
	unsigned   radix;
	unsigned   fito_vec_len;
	unsigned   foti_vec_len;

	void     (*inner)(float *work, unsigned nfft, unsigned lfft, const float *twid);
	void     (*inner_stock)(const float *in, float *out, const float *twid, unsigned ncol, unsigned nrow_div_radix);
	void     (*dif)(float *work, unsigned nfft, unsigned lfft, const float *twid);
	void     (*dit)(float *work, unsigned nfft, unsigned lfft, const float *twid);
	void     (*stock)(const float *in, float *out, const float *twid, unsigned ncol, unsigned nrow_div_radix);
};


#define SENTINAL_PASS {0, 0, 0, NULL, NULL, NULL, NULL, NULL}
#define FLOAT_PASS_EVERY(vtyp_, radix_, vwidth_, foti_width_) \
{   radix_ \
,   vwidth_ \
,   foti_width_ \
,   fftset_ ## vtyp_ ## _r ## radix_ ## _inner \
,   fftset_ ## vtyp_ ## _r ## radix_ ## _inner_stock \
,   fftset_ ## vtyp_ ## _r ## radix_ ## _dif \
,   fftset_ ## vtyp_ ## _r ## radix_ ## _dit \
,   fftset_ ## vtyp_ ## _r ## radix_ ## _stock \
}
#define FLOAT_PASS_INNER(vtyp_, radix_, vwidth_, foti_width_) \
{   radix_ \
,   vwidth_ \
,   foti_width_ \
,   fftset_ ## vtyp_ ## _r ## radix_ ## _inner \
,   fftset_ ## vtyp_ ## _r ## radix_ ## _inner_stock \
,   NULL \
,   NULL \
,   NULL \
}

/* This list must be sorted by fito_vec_len. */
const struct float_pass_radix FFTSET_FLOAT_PASSES[] =
{FLOAT_PASS_EVERY(v1f, 2,  1, 1)
,FLOAT_PASS_EVERY(v1f, 3,  1, 1)
,FLOAT_PASS_EVERY(v1f, 4,  1, 1)
,FLOAT_PASS_INNER(v1f, 5,  1, 1)
,FLOAT_PASS_INNER(v1f, 6,  1, 1)
,FLOAT_PASS_INNER(v1f, 8,  1, 1)
,FLOAT_PASS_INNER(v1f, 16, 1, 1)
#if V4F_EXISTS
,FLOAT_PASS_EVERY(v4f, 2,  4, 4)
,FLOAT_PASS_EVERY(v4f, 3,  4, 4)
,FLOAT_PASS_EVERY(v4f, 4,  4, 4)
,FLOAT_PASS_INNER(v4f, 5,  4, 4)
,FLOAT_PASS_INNER(v4f, 6,  4, 4)
,FLOAT_PASS_INNER(v4f, 8,  4, 4)
,FLOAT_PASS_INNER(v4f, 16, 4, 4)
#endif
#if V8F_EXISTS
,FLOAT_PASS_EVERY(v8f, 2,  8, 8)
,FLOAT_PASS_EVERY(v8f, 3,  8, 8)
,FLOAT_PASS_EVERY(v8f, 4,  8, 8)
,FLOAT_PASS_INNER(v8f, 5,  8, 8)
,FLOAT_PASS_INNER(v8f, 6,  8, 8)
,FLOAT_PASS_INNER(v8f, 8,  8, 8)
,FLOAT_PASS_INNER(v8f, 16, 8, 8)
#endif
,SENTINAL_PASS
};


#if 0
#if V2D_EXISTS
VECRADIX2PASSES(v2d, V2D, double, 2)
VECRADIX3PASSES(v2d, V2D, double, 2)
VECRADIX4PASSES(v2d, V2D, double, 2)
VECRADIX5PASSES(v2d, V2D, double, 2)
VECRADIX6PASSES(v2d, V2D, double, 2)
VECRADIX8PASSES(v2d, V2D, double, 2)
VECRADIX16PASSES(v2d, V2D, double, 2)
#endif

#if V8F_EXISTS
VECRADIX2PASSES(v8f, V8F, float,  8)
VECRADIX3PASSES(v8f, V8F, float,  8)
VECRADIX4PASSES(v8f, V8F, float,  8)
#endif
#endif



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
	pass = cop_salloc(&(fc->mem), sizeof(*pass), 0);
	if (pass == NULL)
		return NULL;

	/* Detect radix. */
	if (length == 2) {
		pass->twiddle        = NULL;
		pass->lfft_div_radix = 1;
		pass->radix          = 2;
		pass->dif            = fftset_v4f_r2_inner;
		pass->dit            = fftset_v4f_r2_inner;
		pass->dif_stockham   = fftset_v4f_r2_inner_stock;
	} else if (length == 3) {
		pass->twiddle        = NULL;
		pass->lfft_div_radix = 1;
		pass->radix          = 3;
		pass->dif            = fftset_v4f_r3_inner;
		pass->dit            = fftset_v4f_r3_inner;
		pass->dif_stockham   = fftset_v4f_r3_inner_stock;
	} else if (length == 4) {
		pass->twiddle        = NULL;
		pass->lfft_div_radix = 1;
		pass->radix          = 4;
		pass->dif            = fftset_v4f_r4_inner;
		pass->dit            = fftset_v4f_r4_inner;
		pass->dif_stockham   = fftset_v4f_r4_inner_stock;
	} else if (length == 5) {
		pass->twiddle        = NULL;
		pass->lfft_div_radix = 1;
		pass->radix          = 5;
		pass->dif            = fftset_v4f_r5_inner;
		pass->dit            = fftset_v4f_r5_inner;
		pass->dif_stockham   = fftset_v4f_r5_inner_stock;
	} else if (length == 6) {
		pass->twiddle        = NULL;
		pass->lfft_div_radix = 1;
		pass->radix          = 6;
		pass->dif            = fftset_v4f_r6_inner;
		pass->dit            = fftset_v4f_r6_inner;
		pass->dif_stockham   = fftset_v4f_r6_inner_stock;
	} else if (length == 8) {
		pass->twiddle        = NULL;
		pass->lfft_div_radix = 1;
		pass->radix          = 8;
		pass->dif            = fftset_v4f_r8_inner;
		pass->dit            = fftset_v4f_r8_inner;
		pass->dif_stockham   = fftset_v4f_r8_inner_stock;
	} else if (length == 16) {
		pass->twiddle        = NULL;
		pass->lfft_div_radix = 1;
		pass->radix          = 16;
		pass->dif            = fftset_v4f_r16_inner;
		pass->dit            = fftset_v4f_r16_inner;
		pass->dif_stockham   = fftset_v4f_r16_inner_stock;
	} else if (   length % 4 == 0
			  &&  length / 4 != 8
			  &&  length / 4 != 4
			  ) {
		float *twid = cop_salloc(&(fc->mem), sizeof(float) * 6 * length / 4, 64);
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
		pass->dif          = fftset_v4f_r4_dif;
		pass->dit          = fftset_v4f_r4_dit;
		pass->dif_stockham = fftset_v4f_r4_stock;
	} else if (length % 3 == 0) {
		float *twid = cop_salloc(&(fc->mem), sizeof(float) * 4 * length / 3, 64);
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
		pass->dif            = fftset_v4f_r3_dif;
		pass->dit            = fftset_v4f_r3_dit;
		pass->dif_stockham   = fftset_v4f_r3_stock;
	} else if (length % 2 == 0) {
		float *twid = cop_salloc(&(fc->mem), sizeof(float) * length, 64);
		if (twid == NULL)
			return NULL;
		for (i = 0; i < length / 2; i++) {
			twid[2*i+0] = cosf(i * (-(float)M_PI * 2.0f) / length);
			twid[2*i+1] = sinf(i * (-(float)M_PI * 2.0f) / length);
		}
		pass->twiddle      = twid;
		pass->lfft_div_radix = length / 2;
		pass->radix        = 2;
		pass->dif          = fftset_v4f_r2_dif;
		pass->dit          = fftset_v4f_r2_dit;
		pass->dif_stockham = fftset_v4f_r2_stock;
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
