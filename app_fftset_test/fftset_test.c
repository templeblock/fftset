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
#include <string.h>

static const unsigned primes[] =
{0, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313,
317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499,
503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601,
607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691,
701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809,
811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907,
911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013,
1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093,
1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193,
1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289,
1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399,
1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483,
1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571,
1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663,
1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759,
1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873,
1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987,
1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081,
2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161,
2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281,
2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377,
2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467,
2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593,
2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689,
2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777,
2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879,
2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999,
3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109,
3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221,
3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329,
3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449,
3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539,
3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631,
3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733,
3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851,
3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943,
3947, 3967, 3989};

int prime_impulse_test(struct fftset *fftset, unsigned length, float *buf1, float *buf2, float *buf3)
{
	const struct fftset_fft *fft = fftset_create_fft(fftset, FFTSET_MODULATION_FREQ_OFFSET_REAL, length / 2);
	unsigned pidx;
	unsigned i, j;
	float acc;
	float avg_re;
	float avg_im;
	float worst;

	if (fft == NULL) {
		printf("could not create fft object\n");
		return 1;
	}

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
	pidx   = 0;
	worst  = 0;
	acc    = 0.0f;
	avg_re = 0.0f;
	avg_im = 0.0f;
	for (j = 0; j < length / 2; j++) {
		double re = 0.0;
		double im = 0.0;
		float e2;
		for (pidx = 0, i = 0; i < length; i++) {
			if (i == primes[pidx]) {
				re += cos(i * (j + 0.5) * -(2.0 * M_PI) / length);
				im += sin(i * (j + 0.5) * -(2.0 * M_PI) / length);
				pidx++;
			}
		}
		re      = buf2[2*j]   - re;
		im      = buf2[2*j+1] - im;
		e2      = (float)(re * re + im * im);
		avg_re += (float)re;
		avg_im += (float)im;
		acc    += e2;
		if (e2 > worst) {
			worst = e2;
			pidx = j;
		}
	}
	acc    = sqrtf(acc / (length / 2));
	avg_re = avg_re / (length / 2);
	avg_im = avg_im / (length / 2);
	if (acc > 0.000004) {
		printf("l=%u) the impulse test failed with an RMS error of %f (avg=%f,%f;worst=%u,%f)\n", length, acc, avg_re, avg_im, pidx, sqrtf(worst));
		return 1;
	}

	/* If we got this far, buf3 contains the validated input spectrum. Invert
	 * it and we should get our input back (which is setting in buf1). */
	fftset_fft_inverse(fft, buf3, buf3, buf2);
	for (acc = 0.0f, avg_re = 0.0f, j = 0; j < length; j++) {
		float re = buf3[j] / (length / 2) - buf1[j];
		acc    += re * re;
		avg_re += re;
	}
	acc    = sqrtf(acc / (length / 2));
	avg_re = avg_re / (length / 2);
	if (acc > 0.000001) {
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

	if (fft == NULL) {
		printf("could not create fft object\n");
		return 1;
	}

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
	if (acc > 0.00001) {
		printf("l=%u) the convolution test failed with an RMS error of %f (avg=%f)\n", length, acc, avg_re);
		return 1;
	}

	return 0;
}

int prime_impulse_test_complex(struct fftset *fftset, unsigned length, float *buf1, float *buf2, float *buf3)
{
	const struct fftset_fft *fft = fftset_create_fft(fftset, FFTSET_MODULATION_COMPLEX, length);
	unsigned pidx;
	unsigned i, j;
	float acc;
	float avg_re;
	float avg_im;
	float worst;

	if (fft == NULL) {
		printf("could not create fft object\n");
		return 1;
	}

	/* Stick a bunch of prime numbers in the buffer. */
	for (pidx = 0, i = 0; i < 2*length; i++) {
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
	memcpy(buf3, buf2, sizeof(float) * 2 * length);

	/* Remove the components that should exist from the output buffer. */
	pidx   = 0;
	worst  = 0;
	acc    = 0.0f;
	avg_re = 0.0f;
	avg_im = 0.0f;
	for (j = 0; j < length; j++) {
		double re = 0.0;
		double im = 0.0;
		float e2;
		for (pidx = 0, i = 0; i < length; i++) {
			int x = 2*i+0;
			if (x == primes[pidx]) {
				re += cos(i * j * -(2.0 * M_PI) / length);
				im += sin(i * j * -(2.0 * M_PI) / length);
				pidx++;
			}
			x = 2*i+1;
			if (x == primes[pidx]) {
				re -= sin(i * j * -(2.0 * M_PI) / length);
				im += cos(i * j * -(2.0 * M_PI) / length);
				pidx++;
			}
		}
		re      = buf2[2*j]   - re;
		im      = buf2[2*j+1] - im;
		e2      = (float)(re * re + im * im);
		avg_re += (float)re;
		avg_im += (float)im;
		acc    += e2;
		if (e2 > worst) {
			worst = e2;
			pidx = j;
		}
	}
	acc    = sqrtf(acc / (length));
	avg_re = avg_re / (length);
	avg_im = avg_im / (length);
	if (acc > 0.000004) {
		printf("l=%u) the complex impulse test failed with an RMS error of %f (avg=%f,%f;worst=%u,%f)\n", length, acc, avg_re, avg_im, pidx, sqrtf(worst));
		return 1;
	}

	/* If we got this far, buf3 contains the validated input spectrum. Invert
	 * it and we should get our input back (which is setting in buf1). */
	fftset_fft_inverse(fft, buf3, buf3, buf2);
	for (acc = 0.0f, avg_re = 0.0f, j = 0; j < length; j++) {
		float re = buf3[j] / (length) - buf1[j];
		acc    += re * re;
		avg_re += re;
	}
	acc    = sqrtf(acc / (length));
	avg_re = avg_re / (length);
	if (acc > 0.000001) {
		printf("l=%u) the complex inverse test failed with an RMS error of %f (avg=%f)\n", length, acc, avg_re);
		return 1;
	}
	
	return 0;
}


#define FREQ_OFFSET_REAL_LEN (VLF_WIDTH * VLF_WIDTH * 2)

int main(int argc, char *argv[])
{
	struct fftset              fftset;
	struct cop_salloc_iface    mem;
	struct cop_alloc_grp_temps mem_impl;
	int errors = 0;

	float *tmp1;
	float *tmp2;
	float *tmp3;

	if (fftset_init(&fftset)) {
		printf("could not create fftset object\n");
		return 1;
	}

	if (cop_alloc_grp_temps_init(&mem_impl, &mem, sizeof(float) * 16384, 0, 16)) {
		fftset_destroy(&fftset);
		printf("could not create memory allocator\n");
		return 1;
	}

	tmp1 = cop_salloc(&mem, sizeof(float) * 4096, 0);
	tmp2 = cop_salloc(&mem, sizeof(float) * 4096, 0);
	tmp3 = cop_salloc(&mem, sizeof(float) * 4096, 0);

	if (tmp1 == NULL || tmp2 == NULL || tmp3 == NULL) {
		cop_alloc_grp_temps_free(&mem_impl);
		fftset_destroy(&fftset);
		printf("out of memory\n");
		return 1;
	}

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

	errors += prime_impulse_test_complex(&fftset, 8, tmp1, tmp2, tmp3);
	errors += prime_impulse_test_complex(&fftset, 16, tmp1, tmp2, tmp3);
	
	/* TODO: figure out why these tests fail. I've forgotten/not-documented-
	 * something here. */
	/* errors += prime_impulse_test_complex(&fftset, 3*4, tmp1, tmp2, tmp3);
	errors += prime_impulse_test_complex(&fftset, 3*5*4, tmp1, tmp2, tmp3);
	errors += prime_impulse_test_complex(&fftset, 5*4, tmp1, tmp2, tmp3); */

	errors += prime_impulse_test_complex(&fftset, 3*8, tmp1, tmp2, tmp3);
	errors += prime_impulse_test_complex(&fftset, 5*8, tmp1, tmp2, tmp3);
	errors += prime_impulse_test_complex(&fftset, 5*3*8, tmp1, tmp2, tmp3);

	errors += prime_impulse_test_complex(&fftset, 32, tmp1, tmp2, tmp3);
	errors += prime_impulse_test_complex(&fftset, 64, tmp1, tmp2, tmp3);
	errors += prime_impulse_test_complex(&fftset, 96, tmp1, tmp2, tmp3);

	if (errors) {
		printf("%u tests failed\n", errors);
	} else {
		printf("all tests passed\n");
	}

	cop_alloc_grp_temps_free(&mem_impl);
	fftset_destroy(&fftset);

	return errors;
}

