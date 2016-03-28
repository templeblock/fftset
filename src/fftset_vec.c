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

static void fc_v4_dif_r2(float *work_buf, unsigned nfft, unsigned lfft, const float *twid)
{
	unsigned rinc = lfft * 4;
	lfft /= 2;
	do {
		unsigned j;
		for (j = 0; j < lfft; j++, work_buf += 8) {
			v4f nre   = v4f_ld(work_buf + 0);
			v4f nim   = v4f_ld(work_buf + 4);
			v4f fre   = v4f_ld(work_buf + rinc + 0);
			v4f fim   = v4f_ld(work_buf + rinc + 4);
			v4f tre   = v4f_broadcast(twid[2*j+0]);
			v4f tim   = v4f_broadcast(twid[2*j+1]);
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
			v4f_st(work_buf + 0, onre);
			v4f_st(work_buf + 4, onim);
			v4f_st(work_buf + rinc + 0, ofre);
			v4f_st(work_buf + rinc + 4, ofim);
		}
		work_buf += rinc;
	} while (--nfft);
}

static void fc_v4_dif_r4(float *work_buf, unsigned nfft, unsigned lfft, const float *twid)
{
	unsigned rinc = lfft * 2;
	lfft /= 4;
	do {
		unsigned j;
		const float *tp = twid;
		for (j = 0; j < lfft; j++, work_buf += 8, tp += 6) {
			v4f b0r   = v4f_ld(work_buf + 0*rinc + 0);
			v4f b0i   = v4f_ld(work_buf + 0*rinc + 4);
			v4f b1r   = v4f_ld(work_buf + 1*rinc + 0);
			v4f b1i   = v4f_ld(work_buf + 1*rinc + 4);
			v4f b2r   = v4f_ld(work_buf + 2*rinc + 0);
			v4f b2i   = v4f_ld(work_buf + 2*rinc + 4);
			v4f b3r   = v4f_ld(work_buf + 3*rinc + 0);
			v4f b3i   = v4f_ld(work_buf + 3*rinc + 4);
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
			v4f c1r   = v4f_broadcast(tp[0]);
			v4f c1i   = v4f_broadcast(tp[1]);
			v4f c2r   = v4f_broadcast(tp[2]);
			v4f c2i   = v4f_broadcast(tp[3]);
			v4f c3r   = v4f_broadcast(tp[4]);
			v4f c3i   = v4f_broadcast(tp[5]);
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
			v4f_st(work_buf + 0*rinc + 0, tr0);
			v4f_st(work_buf + 0*rinc + 4, ti0);
			v4f_st(work_buf + 1*rinc + 0, o1r);
			v4f_st(work_buf + 1*rinc + 4, o1i);
			v4f_st(work_buf + 2*rinc + 0, o2r);
			v4f_st(work_buf + 2*rinc + 4, o2i);
			v4f_st(work_buf + 3*rinc + 0, o3r);
			v4f_st(work_buf + 3*rinc + 4, o3i);
		}
		work_buf += 3*rinc;
	} while (--nfft);
}

static void fc_v4_dit_r2(float *work_buf, unsigned nfft, unsigned lfft, const float *twid)
{
	unsigned rinc = lfft * 4;
	lfft /= 2;
	do {
		unsigned j;
		for (j = 0; j < lfft; j++, work_buf += 8) {
			v4f nre  = v4f_ld(work_buf + 0);
			v4f nim  = v4f_ld(work_buf + 4);
			v4f ptre = v4f_ld(work_buf + rinc + 0);
			v4f ptim = v4f_ld(work_buf + rinc + 4);
			v4f tre  = v4f_broadcast(twid[2*j+0]);
			v4f tim  = v4f_broadcast(twid[2*j+1]);
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
			v4f_st(work_buf + 0, onre);
			v4f_st(work_buf + 4, onim);
			v4f_st(work_buf + rinc + 0, ofre);
			v4f_st(work_buf + rinc + 4, ofim);
		}
		work_buf += rinc;
	} while (--nfft);
}

static void fc_v4_dit_r4(float *work_buf, unsigned nfft, unsigned lfft, const float *twid)
{
	unsigned rinc = lfft * 2;
	lfft /= 4;
	do {
		unsigned j;
		const float *tp = twid;
		for (j = 0; j < lfft; j++, work_buf += 8, tp += 6) {
			v4f b0r  = v4f_ld(work_buf + 0*rinc + 0);
			v4f b0i  = v4f_ld(work_buf + 0*rinc + 4);
			v4f b1r  = v4f_ld(work_buf + 1*rinc + 0);
			v4f b1i  = v4f_ld(work_buf + 1*rinc + 4);
			v4f b2r  = v4f_ld(work_buf + 2*rinc + 0);
			v4f b2i  = v4f_ld(work_buf + 2*rinc + 4);
			v4f b3r  = v4f_ld(work_buf + 3*rinc + 0);
			v4f b3i  = v4f_ld(work_buf + 3*rinc + 4);
			v4f c1r  = v4f_broadcast(tp[0]);
			v4f c1i  = v4f_broadcast(tp[1]);
			v4f c2r  = v4f_broadcast(tp[2]);
			v4f c2i  = v4f_broadcast(tp[3]);
			v4f c3r  = v4f_broadcast(tp[4]);
			v4f c3i  = v4f_broadcast(tp[5]);
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
			v4f_st(work_buf + 0*rinc + 0, o0r);
			v4f_st(work_buf + 0*rinc + 4, o0i);
			v4f_st(work_buf + 1*rinc + 0, o1r);
			v4f_st(work_buf + 1*rinc + 4, o1i);
			v4f_st(work_buf + 2*rinc + 0, o2r);
			v4f_st(work_buf + 2*rinc + 4, o2i);
			v4f_st(work_buf + 3*rinc + 0, o3r);
			v4f_st(work_buf + 3*rinc + 4, o3i);
		}
		work_buf += 3*rinc;
	} while (--nfft);
}

static void fc_v4_stock_r2(const float *in, float *out, const float *twid, unsigned ncol, unsigned nrow_div_radix)
{
	const unsigned ooffset = (2*4)*ncol;
	const unsigned ioffset = ooffset*nrow_div_radix;
	do {
		const float *in0 = in;
		const float *tp   = twid;
		unsigned     j    = ncol;
		do {
			v4f r0, i0, r1, i1;
			v4f or0, oi0, or1, oi1;
			v4f twr1, twi1;
			V4F_LD2(r0, i0, in0 + 0*ooffset);
			V4F_LD2(r1, i1, in0 + 1*ooffset);
			twr1     = v4f_broadcast(tp[0]);
			twi1     = v4f_broadcast(tp[1]);
			or1      = v4f_sub(r0, r1);
			oi1      = v4f_sub(i0, i1);
			or0      = v4f_add(r0, r1);
			oi0      = v4f_add(i0, i1);
			v4f or1a = v4f_mul(or1, twr1);
			v4f or1b = v4f_mul(oi1, twi1);
			v4f oi1a = v4f_mul(or1, twi1);
			v4f oi1b = v4f_mul(oi1, twr1);
			r1       = v4f_sub(or1a, or1b);
			i1       = v4f_add(oi1a, oi1b);
			v4f_st(out + 0*ioffset+0*4, or0);
			v4f_st(out + 0*ioffset+1*4, oi0);
			v4f_st(out + 1*ioffset+0*4, r1);
			v4f_st(out + 1*ioffset+1*4, i1);
			tp   += 2;
			out  += (2*4);
			in0  += (2*4);
		} while (--j);
		in = in + 2*ooffset;
	} while (--nrow_div_radix);
}

static void fc_v4_stock_r4(const float *in, float *out, const float *twid, unsigned ncol, unsigned nrow_div_radix)
{
	const unsigned ooffset = (2*4)*ncol;
	const unsigned ioffset = ooffset*nrow_div_radix;
	do {
		const float *in0 = in;
		const float *tp  = twid;
		unsigned     j;
		for (j = 0; j < ncol; j++, tp += 6) {
			v4f b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i;
			v4f y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
			v4f z0r, z0i, z1r, z1i, z2r, z2i, z3r, z3i;
			v4f o1r, o1i, o2r, o2i, o3r, o3i;
			v4f c1r, c1i, c2r, c2i, c3r, c3i;
			v4f o1ra, o1ia, o2ra, o2ia, o3ra, o3ia;
			v4f o1rb, o1ib, o2rb, o2ib, o3rb, o3ib;

			V4F_LD2(b0r, b0i, in0 + 0*ooffset);
			V4F_LD2(b1r, b1i, in0 + 1*ooffset);
			V4F_LD2(b2r, b2i, in0 + 2*ooffset);
			V4F_LD2(b3r, b3i, in0 + 3*ooffset);
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
			c1r  = v4f_broadcast(tp[0]);
			c1i  = v4f_broadcast(tp[1]);
			c2r  = v4f_broadcast(tp[2]);
			c2i  = v4f_broadcast(tp[3]);
			c3r  = v4f_broadcast(tp[4]);
			c3i  = v4f_broadcast(tp[5]);
			o1ra = v4f_mul(z1r, c1r);
			o1rb = v4f_mul(z1i, c1i);
			o1ia = v4f_mul(z1r, c1i);
			o1ib = v4f_mul(z1i, c1r);
			o2ra = v4f_mul(z2r, c2r);
			o2rb = v4f_mul(z2i, c2i);
			o2ia = v4f_mul(z2r, c2i);
			o2ib = v4f_mul(z2i, c2r);
			o3ra = v4f_mul(z3r, c3r);
			o3rb = v4f_mul(z3i, c3i);
			o3ia = v4f_mul(z3r, c3i);
			o3ib = v4f_mul(z3i, c3r);
			o1r  = v4f_sub(o1ra, o1rb);
			o1i  = v4f_add(o1ia, o1ib);
			o2r  = v4f_sub(o2ra, o2rb);
			o2i  = v4f_add(o2ia, o2ib);
			o3r  = v4f_sub(o3ra, o3rb);
			o3i  = v4f_add(o3ia, o3ib);
			V4F_ST2(out + 0*ioffset, z0r, z0i);
			V4F_ST2(out + 1*ioffset, o1r, o1i);
			V4F_ST2(out + 2*ioffset, o2r, o2i);
			V4F_ST2(out + 3*ioffset, o3r, o3i);

			out  += (2*4);
			in0  += (2*4);
		}
		in = in + 4*ooffset;
	} while (--nrow_div_radix);
}

struct fftset_vec *fastconv_get_inner_pass(struct fftset *fc, unsigned length)
{
	struct fftset_vec *pass;
	struct fftset_vec **ipos;
	unsigned i;

	/* Search for the pass. */
	for (pass = fc->first_inner; pass != NULL; pass = pass->next) {
		if (pass->lfft == length && pass->dif != NULL)
			return pass;
		if (pass->lfft < length)
			break;
	}

	/* Create new inner pass. */
	pass = aalloc_alloc(&fc->memory, sizeof(*pass));
	if (pass == NULL)
		return NULL;

	/* Detect radix. */
	if (length % 4 == 0) {
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
		pass->lfft         = length;
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
		pass->lfft         = length;
		pass->radix        = 2;
		pass->dif          = fc_v4_dif_r2;
		pass->dit          = fc_v4_dit_r2;
		pass->dif_stockham = fc_v4_stock_r2;
	} else {
		/* Only support radix-2/4. What are you doing crazy-face? */
		abort();
	}

	/* Make next pass if required */
	if (pass->lfft != pass->radix) {
		assert(pass->lfft % pass->radix == 0);
		pass->next_compat = fastconv_get_inner_pass(fc, pass->lfft / pass->radix);
		if (pass->next_compat == NULL)
			return NULL;
	} else {
		pass->next_compat = NULL;
	}

	/* Insert into list. */
	ipos = &(fc->first_inner);
	while (*ipos != NULL && length < (*ipos)->lfft) {
		ipos = &(*ipos)->next;
	}
	pass->next = *ipos;
	*ipos = pass;

	return pass;
}
