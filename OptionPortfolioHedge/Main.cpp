#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifndef Pi 
#define Pi 3.141592653589793238462643 
#endif

#define sqrt2pi 2.506628 // sqrt(2 * Pi)

enum optionType {
	otEuroCall,
	otEuroPut
};

float rnorm() {
	float u = ((float)rand() / (RAND_MAX)) * 2 - 1;
	float v = ((float)rand() / (RAND_MAX)) * 2 - 1;
	float r = u * u + v * v;
	if (r == 0 || r > 1) return rnorm();
	float c = sqrt(-2 * log(r) / r);
	return u * c;
}

float* B = NULL;

float* MB(size_t n, float T) {

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

float pnorm(float x) {

	float L, K, w;
	/* constants */
	double const a1 = 0.31938153, a2 = -0.356563782, a3 = 1.781477937;
	double const a4 = -1.821255978, a5 = 1.330274429;

	L = fabs(x);
	K = 1.0 / (1.0 + 0.2316419 * L);
	w = 1.0 - 1.0 / sqrt(2 * Pi) * exp(-L *L / 2) * (a1 * K + a2 * K *K + a3 * K*K*K + a4 * K*K*K*K + a5 * K*K*K*K*K);

	if (x < 0){
		w = 1.0 - w;
	}

	return w;
}

float* seq(float a, float b, float n) {

	float *x = (float*)calloc(n, sizeof(float));
	float by = (b - a) / n;


	for (size_t i = 0; i < n; i++)
	{
		x[i] = a + by * i;
	}

	return x;
}

float* bs_d(float T, float t, float S0, float K, float r, float sigma) {

	float sigmaSqrtT = sigma * sqrt(T - t);
	float *d = (float*)calloc(2, sizeof(float));
	d[0] = (log(S0 / K) + (r + 0.5*sigma*sigma) * (T - t)) / (sigmaSqrtT);
	d[1] = d[0] - sigmaSqrtT;

	return d;
}

float preco_bs(float T, float t, float S0, float K, float r, float sigma, optionType opcao) {

	float *d = bs_d(T, t, S0, K, r, sigma);

	switch (opcao)
	{
	case otEuroCall:
		return pnorm(d[0]) * S0 - pnorm(d[1]) * K * exp(-r * (T - t));
		break;
	case otEuroPut:
		return pnorm(-d[1]) * K * exp(-r * (T - t)) - pnorm(-d[0]) * S0;
		break;
	default:
		return 0.0;
		break;
	}

}

float* evolucao_real_bs(float T, float S0, float r, float sigma, size_t n, float mu, float* Tempos) {

	float *W;
	float *S = (float*)calloc(n, sizeof(float));

	W = MB(n, T);
	for (size_t i = 0; i < n; i++)
	{
		S[i] = S0 * exp((mu - r - sigma*sigma / 2) * Tempos[i] + sigma * W[i]); // checar esta formula

	}

	return(S);
}

float hedging_bs(float T, float t, float S0, float K, float r, float sigma, optionType opcao) {

	float *d = bs_d(T, t, S0, K, r, sigma);

	switch (opcao)
	{
	case otEuroCall:
		return pnorm(d[0]);
		break;
	case otEuroPut:
		return -pnorm(-d[0]);
		break;
	default:
		return 0.0;
		break;
	}

}

float payoff_descontado(float T, float r, float ST, float K, optionType opcao) {

	float payoff;

	switch (opcao)
	{
	case otEuroCall:
		payoff = ST - K * exp(-r * T); // checar formula
		break;
	case otEuroPut:
		payoff = K * exp(-r * T) - ST; // checar formula
		break;
	default:
		break;
	}

	if (payoff < 0)
		return 0.0;
	else
		return payoff;

}

size_t which(float* f, size_t n, float x) {

	size_t i = 0;
	while ((f[i] < x) && (i < n))
	{
		i++;
	}

	return i;
}

double f(double x) {
	int i;
	double sum = 0.0;
	for (i = 0; i < 1000; i++) {
		sum += 1.0 / (i + 1.0);
	}
	return 1.0 / (x + 1.0);
}

double fint(double x) {
	return log(x + 1.0);
}

//int main(int argc, char *argv[]){
//	printf("Hello World!\n");
//
//	double a = atof(argv[1]);
//	double b = atof(argv[2]);
//	int numintervalos = atoi(argv[3]);
//	int numthreads = atoi(argv[4]);
//	omp_set_num_threads(numthreads);
//
//	double integral = 0.0;
//	double width = (b - a) / numintervalos;
//#pragma omp parallel
//	{
//		int ID = omp_get_thread_num();
//		int i;
//		for (i = ID; i < numintervalos; i += numthreads) {
//			double begin = a + i * width;
//			double end = begin + width;
//			double middle = (begin + end) / 2.0;
//			double fx = f(middle);
//			double area = fx * width;
//#pragma omp atomic
//			integral += area;
//		}
//	}
//
//	printf("Integral de f(x) de %lf ate %lf eh: %0.15lf\n", a, b, integral);
//	printf("          fint(%lf) - fint(%lf) eh: %0.15lf\n", b, a, fint(b) - fint(a));
//
//	printf("Press [ENTER] to exit.");
//	int c = fgetc(stdin);
//}

int main(int argc, char *argv[]){
	double t1 = omp_get_wtime();

	float T = 1.0;
	size_t M = 1000;
	size_t n = 50;

	float mu = 0.01;
	float S0 = 100;
	float r = 0;
	float sigma = 0.09;
	optionType opcao = otEuroCall;
	float K = 100;
	size_t l = 22;

	float *erro = (float*)calloc(M, sizeof(float));
	float *Tempos = seq(0.0, T, n);
	float preco_opcao = preco_bs(T, 0.0, S0, K, r, sigma, opcao);

	size_t i, j, k, g;
	float *S;
	float *S_int = (float*)calloc(l + 1, sizeof(float));
	float *hedging = (float*)calloc(l, sizeof(float));
	float *tempo = (float*)calloc(l, sizeof(float));
	float Pay_descontado_real;
	float integral, valor_portfolio_final, erroM = 0.0;

	for (i = 0; i < M; i++)
	{
		S = evolucao_real_bs(T, S0, r, sigma, n, mu, Tempos);
		Pay_descontado_real = payoff_descontado(T, r, S[n - 1], K, opcao);

		for (j = 0; j<l; j++) {
			tempo[j] = (float)j / (float)l;
			g = which(Tempos, n, tempo[j]);
			S_int[j] = S[g];
			hedging[j] = hedging_bs(T, tempo[j], S_int[j], K, r, sigma, opcao);
		}
		S_int[l] = S[n - 1];

		integral = 0;
		for (k = 0; k<l; k++) {
			integral += hedging[k] * (S_int[k + 1] - S_int[k]);
		}
		valor_portfolio_final = preco_opcao + integral;
		erro[i] = Pay_descontado_real - valor_portfolio_final;
		erroM += erro[i];
	}

	float hedging_error = erroM / M;

	printf("%f\n", hedging_error);

	free(Tempos);
	free(erro);

	double t2 = omp_get_wtime();
	printf("Time elapsed: %f\n", t2 - t1);

	printf("Press [ENTER] to exit.");
	int c = fgetc(stdin);
}