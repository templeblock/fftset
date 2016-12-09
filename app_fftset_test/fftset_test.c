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

#include "fftset/fftset.h"
#include "cop/cop_vec.h"
#include <math.h>
#include <stdio.h>

static const unsigned primes[] = {0, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997};

int prime_impulse_test(struct fftset *fftset, unsigned length, float *buf1, float *buf2, float *buf3)
{
	const struct fftset_fft *fft = fftset_create_fft(fftset, FFTSET_MODULATION_FREQ_OFFSET_REAL, length / 2);
	unsigned pidx;
	unsigned i, j;
	float acc;
	float avg_re;
	float avg_im;
	float worst;

	/* Stick a bunch of prime numbers in the buffer. */
	for (pidx = 0, i = 0; i < length; i++) {
		if (i == primes[pidx]) {
			buf1[i] = 1.0f;
			pidx++;
		} else {
			buf1[i] = 0.0f;
		}
	}

	/* Do a forward FFT. */
	fftset_fft_forward(fft, buf2, buf1, buf3);

	/* Save it. */
	memcpy(buf3, buf2, sizeof(float) * length);

	/* Remove the components that should exist from the output buffer. */
	for (pidx = 0, i = 0; i < length; i++) {
		if (i == primes[pidx]) {
			for (j = 0; j < length / 2; j++) {
				float re = cosf(i * (j + 0.5f) * -((float)(2.0 * M_PI) / length));
				float im = sinf(i * (j + 0.5f) * -((float)(2.0 * M_PI) / length));
				buf2[2*j]   -= re;
				buf2[2*j+1] -= im;
			}
			pidx++;
		}
	}

	/* Sum power of remaining buffer. */
	pidx = 0;
	worst = 0;
	for (acc = 0.0f, avg_re = 0.0f, avg_im = 0.0f, j = 0; j < length / 2; j++) {
		float re = buf2[2*j];
		float im = buf2[2*j+1];
		float e2 = re * re + im * im;
		acc += e2;
		if (e2 > worst) {
			worst = e2;
			pidx = j;
		}
		avg_re += re;
		avg_im += im;
	}
	acc    = sqrtf(acc / (length / 2));
	avg_re = avg_re / (length / 2);
	avg_im = avg_im / (length / 2);
	if (acc > 0.0002) {
		printf("l=%u) the impulse test failed with an RMS error of %f (avg=%f,%f;worst=%u,%f)\n", length, acc, avg_re, avg_im, pidx, sqrtf(worst));
		return 1;
	}

	/* If we got this far, buf3 contains the validated input spectrum. Invert
	 * it and we should get our input back (which is setting in buf1). */
	fftset_fft_inverse(fft, buf3, buf3, buf2);
	for (acc = 0.0f, avg_re = 0.0f, j = 0; j < length; j++) {
		float re = buf3[j] - buf1[j] * (length / 2);
		acc    += re * re;
		avg_re += re;
	}
	acc    = sqrtf(acc / (length / 2));
	avg_re = avg_re / (length / 2);
	if (acc > 0.00003) {
		printf("l=%u) the inverse test failed with an RMS error of %f (avg=%f)\n", length, acc, avg_re);
		return 1;
	}

	return 0;
}

int convolution_test(struct fftset *fftset, unsigned length, float *buf1, float *buf2, float *buf3)
{
	const struct fftset_fft *fft = fftset_create_fft(fftset, FFTSET_MODULATION_FREQ_OFFSET_REAL, length / 2);
	unsigned pidx;
	unsigned i, j;
	float acc;
	float avg_re;

	/* Stick a bunch of prime numbers in the first quarter of the buffer. */
	for (pidx = 0, i = 0; i < length; i++) {
		if (i == primes[pidx] && i < length / 4) {
			buf1[i] = 2.0f / length;
			pidx++;
		} else {
			buf1[i] = 0.0f;
		}
	}

	/* Put the kernel into buf2. */
	fftset_fft_conv_get_kernel(fft, buf2, buf1);

	/* More primes. */
	for (pidx = 0, i = 0; i < length; i++) {
		if (i == primes[pidx] && i < (length * 3) / 4) {
			buf1[i] = 1.0f;
			pidx++;
		} else {
			buf1[i] = 0.0f;
		}
	}

	/* Convolve. */
	fftset_fft_conv(fft, buf1, buf1, buf2, buf3);

	/* Build result. */
	for (i = 0; i < length; i++)
		buf3[i] = 0.0f;
	for (i = 0; primes[i] < (length * 3) / 4; i++) {
		for (j = 0; primes[j] < length / 4; j++) {
			buf3[primes[i]+primes[j]] += 1.0f;
		}
	}

	for (acc = 0.0f, avg_re = 0.0f, j = 0; j < length; j++) {
		float re = buf3[j] - buf1[j];
		acc    += re * re;
		avg_re += re;
	}
	acc    = sqrtf(acc / (length / 2));
	avg_re = avg_re / (length / 2);
	if (acc > 0.00003) {
		printf("l=%u) the convolution test failed with an RMS error of %f (avg=%f)\n", length, acc, avg_re);
		return 1;
	}

	return 0;
}

#define FREQ_OFFSET_REAL_LEN (VLF_WIDTH * VLF_WIDTH * 2)

int main(int argc, char *argv[])
{
	struct fftset fftset;
	struct aalloc aalloc;
	int errors = 0;

	float *tmp1;
	float *tmp2;
	float *tmp3;

	aalloc_init(&aalloc, 33554432, 64, 16384);
	fftset_init(&fftset);

	tmp1 = aalloc_alloc(&aalloc, sizeof(float) * 1024);
	tmp2 = aalloc_alloc(&aalloc, sizeof(float) * 1024);
	tmp3 = aalloc_alloc(&aalloc, sizeof(float) * 1024);

	/* Test passthrough support. */
	errors += prime_impulse_test(&fftset, FREQ_OFFSET_REAL_LEN,  tmp1, tmp2, tmp3);
	errors += convolution_test(&fftset, FREQ_OFFSET_REAL_LEN,  tmp1, tmp2, tmp3);

	/* Test radix-2 most-inner pass. */
	errors += prime_impulse_test(&fftset, FREQ_OFFSET_REAL_LEN * 2,  tmp1, tmp2, tmp3);
	errors += convolution_test(&fftset, FREQ_OFFSET_REAL_LEN * 2,  tmp1, tmp2, tmp3);

	/* Test radix-3 most-inner pass. */
	errors += prime_impulse_test(&fftset, FREQ_OFFSET_REAL_LEN * 3,  tmp1, tmp2, tmp3);
	errors += convolution_test(&fftset, FREQ_OFFSET_REAL_LEN * 3,  tmp1, tmp2, tmp3);

	/* Test radix-4 most-inner pass. */
	errors += prime_impulse_test(&fftset, FREQ_OFFSET_REAL_LEN * 4,  tmp1, tmp2, tmp3);
	errors += convolution_test(&fftset, FREQ_OFFSET_REAL_LEN * 4,  tmp1, tmp2, tmp3);

	/* Test radix-5 most-inner pass. */
	errors += prime_impulse_test(&fftset, FREQ_OFFSET_REAL_LEN * 5,  tmp1, tmp2, tmp3);
	errors += convolution_test(&fftset, FREQ_OFFSET_REAL_LEN * 5,  tmp1, tmp2, tmp3);

	/* Test radix-6 most-inner pass. */
	errors += prime_impulse_test(&fftset, FREQ_OFFSET_REAL_LEN * 6,  tmp1, tmp2, tmp3);
	errors += convolution_test(&fftset, FREQ_OFFSET_REAL_LEN * 6,  tmp1, tmp2, tmp3);

	/* Test radix-8 most-inner pass. */
	errors += prime_impulse_test(&fftset, FREQ_OFFSET_REAL_LEN * 8,  tmp1, tmp2, tmp3);
	errors += convolution_test(&fftset, FREQ_OFFSET_REAL_LEN * 8,  tmp1, tmp2, tmp3);

	/* Test radix-16 most-inner pass. */
	errors += prime_impulse_test(&fftset, FREQ_OFFSET_REAL_LEN * 16,  tmp1, tmp2, tmp3);
	errors += convolution_test(&fftset, FREQ_OFFSET_REAL_LEN * 16,  tmp1, tmp2, tmp3);

	/* Test radix-2 vector passes. */
	errors += prime_impulse_test(&fftset, FREQ_OFFSET_REAL_LEN * 3 * 2,  tmp1, tmp2, tmp3);
	errors += convolution_test(&fftset, FREQ_OFFSET_REAL_LEN * 3 * 2,  tmp1, tmp2, tmp3);

	/* Test radix-3 vector pass. */
	errors += prime_impulse_test(&fftset, FREQ_OFFSET_REAL_LEN * 3 * 3,  tmp1, tmp2, tmp3);
	errors += convolution_test(&fftset, FREQ_OFFSET_REAL_LEN * 3 * 3,  tmp1, tmp2, tmp3);

	/* Test radix-4 vector passes. */
	errors += prime_impulse_test(&fftset, FREQ_OFFSET_REAL_LEN * 3 * 4,  tmp1, tmp2, tmp3);
	errors += convolution_test(&fftset, FREQ_OFFSET_REAL_LEN * 3 * 4,  tmp1, tmp2, tmp3);

	if (errors) {
		printf("%u tests failed\n", errors);
	} else {
		printf("all tests passed\n");
	}

	fftset_destroy(&fftset);
	aalloc_free(&aalloc);

	return errors;
}

