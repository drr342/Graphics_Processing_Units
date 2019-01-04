/*
 *  SEQUENTIAL PRIME NUMBERS GENERATOR
 *
 *  Last name: RIVERA RUIZ
 *  First name: DANIEL
 *  Net ID: drr342
 *
 *  Compile and run in cuda2 with:
 *  gcc -o seqgenprimes -std=c99 seqgenprimes.c
 *  ./seqgenprimes N
 *
 */

#include <stdlib.h>
#include <stdio.h>

void calculate(int *, int);
void toFile(int *, int);

int main(int argc, char * argv[]) {
    int N = atoi(argv[1]);
    int * primes = (int*) calloc(N - 1, sizeof(int));

    calculate(primes, N);
    toFile(primes, N);

    free(primes);
    return 0;
}

void calculate(int * primes, int N) {
    int stop = (int) (N + 1.0) / 2.0;
    for (int i = 2; i <= stop; i++) {
        if (primes[i - 2] == 0) {
            int mult = 2 * i;
            while (mult <= N) {
                primes[mult - 2] = 1;
                mult += i;
            }
        }
    }
}

void toFile(int * primes, int N) {
    char * fileName = (char*) malloc(13 * sizeof(char));
    sprintf(fileName, "%d.txt", N);
    FILE * fp;
    fp = fopen(fileName,"w");
    for (int i = 0; i < N - 1; i++) {
        if (primes[i] == 0)
            fprintf(fp, "%d ", i + 2);
    }
    fprintf(fp, "\n");
    fclose(fp);
    free(fileName);
}
