#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

float rnorm() {
	float u = ((float)rand() / (RAND_MAX)) * 2 - 1;
	float v = ((float)rand() / (RAND_MAX)) * 2 - 1;
	float r = u * u + v * v;
	if (r == 0 || r > 1) return rnorm();
	float c = sqrt(-2 * log(r) / r);
	return u * c;
}

float *B = NULL;

float *MB(size_t n, float T) {

	float h = T / n;
	float sh = sqrt(h);
	float r;

	if (B == NULL)
		B = (float *)calloc(n + 1, sizeof(float));
	size_t i;
	for (i = 0; i < n; i++)
	{
		r = rnorm() * sh;
		B[i + 1] = B[i] + r;
	}

	return B;
}

//
//int main() {
//
//	double t1 = omp_get_wtime();
//	srand(0);
//
//	float T = 1.0;
//
//	size_t i;
//	size_t n = 5000;
//	size_t m = 5000;
//
//	float *B;
//	for (i = 0; i < m; i++)
//	{
//		B = MB(n, T);
//	}
//
//	double t2 = omp_get_wtime();
//
//	printf("%f\n", t2 - t1);
//
//	return 0;
//}