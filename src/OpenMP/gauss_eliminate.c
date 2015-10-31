/* Gaussian elimination code.
* Author: Naga Kandasamy, 10/24/2015
*
* Compile as follows: 
* gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -fopenmp -std=c99 -O3 -lm
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50
#define NUM_THREADS 16
#define TRUE 1
#define FALSE !TRUE

extern int compute_gold(float*, const float*, unsigned int);
Matrix allocate_matrix(int num_rows, int num_columns, int init);
void gauss_eliminate_using_openmp(const Matrix, Matrix);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, unsigned int, float);
int testing = TRUE;
static int num_of_threads = 16;
//static int num_elements = MATRIX_SIZE;
int main(int argc, char** argv) 
{
	if ( !testing )
	{
		num_of_threads = NUM_THREADS;
		if ( argc > 1 )
		{
			printf("Error. This program accepts no arguments.\n");
			exit(0);
		}	
	}
	else
	{
		if ( argc != 2 )
		{
			printf("Input num of threads. \n");
			exit(0);
		}
		else
		{
			num_of_threads = atoi(argv[1]);
			printf("Using %d number of threads.", num_of_threads);
		}	
	}
	/* Allocate and initialize the matrices. */
	Matrix  A;                                              /* The N x N input matrix. */
	Matrix  U;                                              /* The upper triangular matrix to be computed. */

	srand(time(NULL));

	A = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1);      /* Create a random N x N matrix. */
	U = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);      /* Create a random N x 1 vector. */

	/* Gaussian elimination using the reference code. */
	Matrix reference = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);
	struct timeval start, stop;	
	gettimeofday(&start, NULL);

	printf("Performing gaussian elimination using the reference code. \n");
	int status = compute_gold(reference.elements, A.elements, A.num_rows);

	gettimeofday(&stop, NULL);
	printf("CPU run time = %0.5f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	if ( status == 0 )
	{
		printf("Failed to convert given matrix to upper triangular. Try again. Exiting. \n");
		exit(0);
	}
	status = perform_simple_check(reference); // Check that the principal diagonal elements are 1 
	if ( status == 0 )
	{
		printf("The upper triangular matrix is incorrect. Exiting. \n");
		exit(0); 
	}
	printf("Gaussian elimination using the reference code was successful. \n");

	/* WRITE THIS CODE: Perform the Gaussian elimination using the multi-threaded OpenMP version. 
	* The resulting upper triangular matrix should be returned in U
	* */
	gettimeofday(&start, NULL);
	gauss_eliminate_using_openmp(A, U);
	gettimeofday(&stop, NULL);
	printf("CPU run time = %0.5f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	
	/* check if the OpenMP result is equivalent to the expected solution. */
	int size = MATRIX_SIZE*MATRIX_SIZE;
	int res = check_results(reference.elements, U.elements, size, 0.001f);
	printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

	free(A.elements); A.elements = NULL;
	free(U.elements); U.elements = NULL;
	free(reference.elements); reference.elements = NULL;

	return 0;
}


/* Write code to perform gaussian elimination using OpenMP. */

void gauss_eliminate_using_openmp(const Matrix matA, Matrix matU)                 
{
	unsigned int i, j, k;
	unsigned int num_elements = matA.num_columns;
	float *A = matA.elements, *U = matU.elements;
	if ( !A || !U )
	{
		printf("Invalid inputs.\n");
		return;
	}
#pragma omp parallel private(i, j, k,) shared(U, A, num_elements) num_threads(num_of_threads)
{
	/* Copy the contents of the A matrix into the U matrix.*/
#pragma omp for 
    for ( i = 0; i < num_elements; i++ )
        for ( j = 0; j < num_elements; j++ )
            U[num_elements * i + j] = A[num_elements*i + j];		

    for ( k = 0; k < num_elements; k++ )
	{          
	/* Perform Gaussian elimination in place on the U matrix. */
#pragma omp for
        for ( j = (k + 1); j < num_elements; j++ )
		{   /* Reduce the current row. */
			if ( U[num_elements*k + k] == 0 )
			{
				printf("Numerical instability detected. The principal diagonal element is zero.\n");				
			}
            /* Division step.*/
			U[num_elements * k + j] = (float)(U[num_elements * k + j] / U[num_elements * k + k]);
        }

#pragma omp master
        U[num_elements * k + k] = 1; /* Set the principal diagonal entry in U to be 1.*/

#pragma omp for
		for ( i = (k+1); i < num_elements; i++ )
		{
            for ( j = (k+1); j < num_elements; j++ )
                /* Elimnation step. */
				U[num_elements * i + j] = U[num_elements * i + j] -
                                          (U[num_elements * i + k] * U[num_elements * k + j]);

            U[num_elements * i + k] = 0; 
		}
}
	}
}


/* Check if refernce results match multi threaded results. */
int check_results(float *A, float *B, unsigned int size, float tolerance)   
{
	for ( int i = 0; i < size; i++ )
		if ( fabsf(A[i] - B[i]) > tolerance )
			return 0;
	return 1;
}


/* Allocate a matrix of dimensions height*width. 
* If init == 0, initialize to all zeroes.  
* If init == 1, perform random initialization.
* */
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
	Matrix M;
	M.num_columns = M.pitch = num_columns;
	M.num_rows = num_rows;
	int size = M.num_rows * M.num_columns;
	M.elements = (float*) malloc(size*sizeof(float));

	for ( unsigned int i = 0; i < size; i++ )
	{
		if ( init == 0 ) 
			M.elements[i] = 0; 
		else
			M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
	return M;
}	


 /* Returns a random FP number between min and max values. */
float get_random_number(int min, int max)
{                                   
	return (float)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX)));
}


/* Check for upper triangular matrix, that is, the principal diagonal elements are 1. */
int perform_simple_check(const Matrix M)
{                                  
	for ( unsigned int i = 0; i < M.num_rows; i++ )
		if ( (fabs(M.elements[M.num_rows*i + i] - 1.0)) > 0.001) 
			return 0;
	return 1;
} 


