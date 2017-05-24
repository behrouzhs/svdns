/*
Spectral Word Embedding with Negative Sampling (SVD-NS)
The SVD code is taken from SVDLIBC (with the following copyright) 
SVDLIBC is based on ATLAS library which is a fast linear algebra toolkit.

SVDLIBC Copyright
Copyright © 2002, University of Tennessee Research Foundation.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the University of Tennessee nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <float.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <omp.h>
#include <time.h>

#ifdef _WIN32
#define _POPEN_ _popen
#define _PCLOSE_ _pclose
double timer()
{
	return ((double)((double)clock() / (double)CLOCKS_PER_SEC));
}
#else
#define _POPEN_ popen
#define _PCLOSE_ pclose
double timer()
{
	struct timespec now;
	clock_gettime(CLOCK_REALTIME, &now);
	return ((double)((double)now.tv_sec + ((double)now.tv_nsec * 1.0e-9)));
}
#endif

#define _FILE_OFFSET_BITS 64

#ifndef FALSE
#  define FALSE 0
#endif
#ifndef TRUE
#  define TRUE  1
#endif
#define SAFE_FREE(a) {if (a) {free(a); a = NULL;}}
#define BUNZIP2  "bzip2 -d"
#define BZIP2    "bzip2 -1"
#define UNZIP    "gzip -d"
#define ZIP      "gzip -1"
#define COMPRESS "compress"

#define MAX_FILENAME 512
#define MAX_PIPES    64

#define MAXLL 2
#define LMTNW   100000000 /* max. size of working area allowed  */

#define DEFAULT_DIM 100
#define DEFAULT_THREAD 1
#define DEFAULT_SHIFT 0.0
#define DEFAULT_CUTP -DBL_MAX
#define SORT_FUNC NS_compare_colloc_rowmajor
//#define SORT_FUNC NS_compare_colloc_colmajor

/******************************** Structures *********************************/
typedef struct smat *SMat;
typedef struct dmat *DMat;
typedef struct svdrec *SVDRec;

/* Harwell-Boeing sparse matrix. */
struct smat {
  long rows;
  long cols;
  long vals;     /* Total non-zero entries. */
  long *pointr;  /* For each col (plus 1), index of first non-zero entry. */
  long *rowind;  /* For each nz entry, the row index. */
  double *value; /* For each nz entry, the value. */
};

/* Row-major dense matrix.  Rows are consecutive vectors. */
struct dmat {
  long rows;
  long cols;
  double **value; /* Accessed by [row][col]. Free value[0] and value to free.*/
};

struct svdrec {
  int d;      /* Dimensionality (rank) */
  DMat Ut;    /* Transpose of left singular vectors. (d by m)
                 The vectors are the rows of Ut. */
  double *S;  /* Array of singular values. (length d) */
  DMat Vt;    /* Transpose of right singular vectors. (d by n)
                 The vectors are the rows of Vt. */
};

enum storeVals {STORQ = 1, RETRQ, STORP, RETRP};
enum svdCounters {SVD_MXV, SVD_COUNTERS};
enum svdFileFormats {SVD_F_STH, SVD_F_ST, SVD_D_SB, SVD_F_SB, SVD_F_DT, SVD_V_DT, SVD_F_DB};

/* True if a file format is sparse: */
#define SVD_IS_SPARSE(format) ((format >= SVD_F_STH) && (format <= SVD_F_SB))

static char *error_msg[] = {  /* error messages used by function    *
                               * check_parameters                   */
  NULL,
  "",
  "ENDL MUST BE LESS THAN ENDR",
  "REQUESTED DIMENSIONS CANNOT EXCEED NUM ITERATIONS",
  "ONE OF YOUR DIMENSIONS IS LESS THAN OR EQUAL TO ZERO",
  "NUM ITERATIONS (NUMBER OF LANCZOS STEPS) IS INVALID",
  "REQUESTED DIMENSIONS (NUMBER OF EIGENPAIRS DESIRED) IS INVALID",
  "6*N+4*ITERATIONS+1 + ITERATIONS*ITERATIONS CANNOT EXCEED NW",
  "6*N+4*ITERATIONS+1 CANNOT EXCEED NW", NULL};

double **LanStore, *OPBTemp;
double eps, eps1, reps, eps34;
long ierr;
static FILE *Pipe[MAX_PIPES];
static int numPipes = 0;

typedef struct collocation {
	int row;
	int col;
	double val;
} COLLOC;

char *SVDVersion = "1.4";
long SVDVerbosity = 1;
long SVDCount[SVD_COUNTERS];

int no_thread = DEFAULT_THREAD;

void svdResetCounters(void);
void   purge(long n, long ll, double *r, double *q, double *ra,  
             double *qa, double *wrk, double *eta, double *oldeta, long step, 
             double *rnmp, double tol);
void   ortbnd(double *alf, double *eta, double *oldeta, double *bet, long step,
              double rnm);
double startv(SMat A, double *wptr[], long step, long n);
void   store(long, long, long, double *);
void   imtql2(long, long, double *, double *, double *);
void   imtqlb(long n, double d[], double e[], double bnd[]);
void   write_header(long, long, double, double, long, double, long, long, 
                    long);
long   check_parameters(SMat A, long dimensions, long iterations, 
                        double endl, double endr, long vectors);
int    lanso(SMat A, long iterations, long dimensions, double endl,
             double endr, double *ritz, double *bnd, double *wptr[], 
             long *neigp, long n);
long   ritvec(long n, SMat A, SVDRec R, double kappa, double *ritz, 
              double *bnd, double *alf, double *bet, double *w2, 
              long steps, long neig);
long   lanczos_step(SMat A, long first, long last, double *wptr[],
                    double *alf, double *eta, double *oldeta,
                    double *bet, long *ll, long *enough, double *rnmp, 
                    double *tolp, long n);
void   stpone(SMat A, double *wrkptr[], double *rnmp, double *tolp, long n);
long   error_bound(long *, double, double, double *, double *, long step, 
                   double tol);
void   machar(long *ibeta, long *it, long *irnd, long *machep, long *negep);

/* Allocates an array of longs. */
long *svd_longArray(long size, char empty, char *name);
/* Allocates an array of doubles. */
double *svd_doubleArray(long size, char empty, char *name);

void svd_debug(char *fmt, ...);
void svd_error(char *fmt, ...);
void svd_fatalError(char *fmt, ...);
FILE *svd_fatalReadFile(char *filename);
FILE *svd_readFile(char *fileName);
FILE *svd_writeFile(char *fileName, char append);
void svd_closeFile(FILE *file);

/************************************************************** 
 * returns |a| if b is positive; else fsign returns -|a|      *
 **************************************************************/ 
double svd_fsign(double a, double b);

/************************************************************** 
 * returns the larger of two double precision numbers         *
 **************************************************************/ 
double svd_dmax(double a, double b);

/************************************************************** 
 * returns the smaller of two double precision numbers        *
 **************************************************************/ 
double svd_dmin(double a, double b);

/************************************************************** 
 * returns the larger of two integers                         *
 **************************************************************/ 
long svd_imax(long a, long b);

/************************************************************** 
 * returns the smaller of two integers                        *
 **************************************************************/ 
long svd_imin(long a, long b);

/************************************************************** 
 * Function scales a vector by a constant.     		      *
 * Based on Fortran-77 routine from Linpack by J. Dongarra    *
 **************************************************************/ 
void svd_dscal(long n, double da, double *dx, long incx);

/************************************************************** 
 * function scales a vector by a constant.	     	      *
 * Based on Fortran-77 routine from Linpack by J. Dongarra    *
 **************************************************************/ 
void svd_datx(long n, double da, double *dx, long incx, double *dy, long incy);

/************************************************************** 
 * Function copies a vector x to a vector y	     	      *
 * Based on Fortran-77 routine from Linpack by J. Dongarra    *
 **************************************************************/ 
void svd_dcopy(long n, double *dx, long incx, double *dy, long incy);

/************************************************************** 
 * Function forms the dot product of two vectors.      	      *
 * Based on Fortran-77 routine from Linpack by J. Dongarra    *
 **************************************************************/ 
double svd_ddot(long n, double *dx, long incx, double *dy, long incy);

/************************************************************** 
 * Constant times a vector plus a vector     		      *
 * Based on Fortran-77 routine from Linpack by J. Dongarra    *
 **************************************************************/ 
void svd_daxpy (long n, double da, double *dx, long incx, double *dy, long incy);

/********************************************************************* 
 * Function sorts array1 and array2 into increasing order for array1 *
 *********************************************************************/
void svd_dsort2(long igap, long n, double *array1, double *array2);

/************************************************************** 
 * Function interchanges two vectors		     	      *
 * Based on Fortran-77 routine from Linpack by J. Dongarra    *
 **************************************************************/ 
void svd_dswap(long n, double *dx, long incx, double *dy, long incy);

/***************************************************************** 
 * Function finds the index of element having max. absolute value*
 * based on FORTRAN 77 routine from Linpack by J. Dongarra       *
 *****************************************************************/ 
long svd_idamax(long n, double *dx, long incx);

/**************************************************************
 * multiplication of matrix B by vector x, where B = A'A,     *
 * and A is nrow by ncol (nrow >> ncol). Hence, B is of order *
 * n = ncol (y stores product vector).		              *
 **************************************************************/
void svd_opb(SMat A, double *x, double *y, double *temp);

/***********************************************************
 * multiplication of matrix A by vector x, where A is 	   *
 * nrow by ncol (nrow >> ncol).  y stores product vector.  *
 ***********************************************************/
void svd_opa(SMat A, double *x, double *y);

/***********************************************************************
 *                                                                     *
 *				random2()                              *
 *                        (double precision)                           *
 ***********************************************************************/
double svd_random2(long *iy);

/************************************************************** 
 *							      *
 * Function finds sqrt(a^2 + b^2) without overflow or         *
 * destructive underflow.				      *
 *							      *
 **************************************************************/ 
double svd_pythag(double a, double b);

/* Creates an empty dense matrix. */
DMat svdNewDMat(int rows, int cols);
/* Frees a dense matrix. */
void svdFreeDMat(DMat D);

/* Creates an empty sparse matrix. */
SMat svdNewSMat(int rows, int cols, int vals);
/* Frees a sparse matrix. */
void svdFreeSMat(SMat S);

/* Creates an empty SVD record. */
SVDRec svdNewSVDRec(void);
/* Frees an svd rec and all its contents. */
void svdFreeSVDRec(SVDRec R);

/* Transposes a sparse matrix (returning a new one) */
SMat svdTransposeS(SMat S);

/* Writes an array to a file. */
void svdWriteDenseArray(double *a, int n, char *filename);

/* Writes a dense matrix to a file in a given format. */
void svdWriteDenseMatrix(DMat A, char *filename);

/* Performs the las2 SVD algorithm and returns the resulting Ut, S, and Vt. */
SVDRec svdLAS2(SMat A, long dimensions, long iterations, double end[2], 
                      double kappa);
/* Chooses default parameter values.  Set dimensions to 0 for all dimensions: */
SVDRec svdLAS2A(SMat A, long dimensions);





long *svd_longArray(long size, char empty, char *name) {
  long *a;
  if (empty) a = (long *) calloc(size, sizeof(long));
  else a = (long *) malloc(size * sizeof(long));
  if (!a) {
    perror(name);
    /* exit(errno); */
  }
  return a;
}

double *svd_doubleArray(long size, char empty, char *name) {
  double *a;
  if (empty) a = (double *) calloc(size, sizeof(double));
  else a = (double *) malloc(size * sizeof(double));
  if (!a) {
    perror(name);
    /* exit(errno); */
  }
  return a;
}

void svd_beep(void) {
  fputc('\a', stderr);
  fflush(stderr);
}

void svd_debug(char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
}

void svd_error(char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  svd_beep();
  fprintf(stderr, "ERROR: ");
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
  va_end(args);
}

void svd_fatalError(char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  svd_beep();
  fprintf(stderr, "ERROR: ");
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\a\n");
  va_end(args);
  exit(1);
}

static char isPipe(FILE *p) {
  int i;
  for (i = 0; i < numPipes && Pipe[i] != p; i++);
  if (i == numPipes) return FALSE;
  Pipe[i] = Pipe[--numPipes];
  return TRUE;
}

FILE *svd_fatalReadFile(char *filename) {
  FILE *file;
  if (!(file = svd_readFile(filename)))
    svd_fatalError("couldn't read the file %s", filename);
  return file;
}

/* Will silently return NULL if file couldn't be opened */
FILE *svd_readFile(char *fileName) {
  struct stat statbuf;

  /* Try just opening normally */
  if (!stat(fileName, &statbuf))
    return fopen(fileName, "rb");
  
  return NULL;
}

FILE *svd_writeFile(char *fileName, char append) {
  /* Special file name */
  if (!strcmp(fileName, "-"))
    return stdout;
  return (append) ? fopen(fileName, "a") : fopen(fileName, "w");
}

/* Could be a file or a stream. */
void svd_closeFile(FILE *file) {
  if (file == stdin || file == stdout) return;
  if (isPipe(file)) _PCLOSE_(file);
  else fclose(file);
}


/************************************************************** 
 * returns |a| if b is positive; else fsign returns -|a|      *
 **************************************************************/ 
double svd_fsign(double a, double b) {
  if ((a>=0.0 && b>=0.0) || (a<0.0 && b<0.0))return(a);
  else return -a;
}

/************************************************************** 
 * returns the larger of two double precision numbers         *
 **************************************************************/ 
double svd_dmax(double a, double b) {
   return (a > b) ? a : b;
}

/************************************************************** 
 * returns the smaller of two double precision numbers        *
 **************************************************************/ 
double svd_dmin(double a, double b) {
  return (a < b) ? a : b;
}

/************************************************************** 
 * returns the larger of two integers                         *
 **************************************************************/ 
long svd_imax(long a, long b) {
  return (a > b) ? a : b;
}

/************************************************************** 
 * returns the smaller of two integers                        *
 **************************************************************/ 
long svd_imin(long a, long b) {
  return (a < b) ? a : b;
}

/************************************************************** 
 * Function scales a vector by a constant.     		      *
 * Based on Fortran-77 routine from Linpack by J. Dongarra    *
 **************************************************************/ 
void svd_dscal(long n, double da, double *dx, long incx) {
  long i;
  
  if (n <= 0 || incx == 0) return;
  if (incx < 0) dx += (-n+1) * incx;
  for (i=0; i < n; i++) {
    *dx *= da;
    dx += incx;
  }
  return;
}

/************************************************************** 
 * function scales a vector by a constant.	     	      *
 * Based on Fortran-77 routine from Linpack by J. Dongarra    *
 **************************************************************/ 
void svd_datx(long n, double da, double *dx, long incx, double *dy, long incy) {
  long i;
  
  if (n <= 0 || incx == 0 || incy == 0 || da == 0.0) return;
  if (incx == 1 && incy == 1) 
    for (i=0; i < n; i++) *dy++ = da * (*dx++);
  
  else {
    if (incx < 0) dx += (-n+1) * incx;
    if (incy < 0) dy += (-n+1) * incy;
    for (i=0; i < n; i++) {
      *dy = da * (*dx);
      dx += incx;
      dy += incy;
    }
  }
  return;
}

/************************************************************** 
 * Function copies a vector x to a vector y	     	      *
 * Based on Fortran-77 routine from Linpack by J. Dongarra    *
 **************************************************************/ 
void svd_dcopy(long n, double *dx, long incx, double *dy, long incy) {
  long i;
  
  if (n <= 0 || incx == 0 || incy == 0) return;
  if (incx == 1 && incy == 1) 
    for (i=0; i < n; i++) *dy++ = *dx++;
  
  else {
    if (incx < 0) dx += (-n+1) * incx;
    if (incy < 0) dy += (-n+1) * incy;
    for (i=0; i < n; i++) {
      *dy = *dx;
      dx += incx;
      dy += incy;
    }
  }
  return;
}

/************************************************************** 
 * Function forms the dot product of two vectors.      	      *
 * Based on Fortran-77 routine from Linpack by J. Dongarra    *
 **************************************************************/ 
double svd_ddot(long n, double *dx, long incx, double *dy, long incy) {
	long i;
	double dot_product;

	if (n <= 0 || incx == 0 || incy == 0) return(0.0);
	dot_product = 0.0;
	if (incx == 1 && incy == 1)
		for (i = 0; i < n; i++) dot_product += (*dx++) * (*dy++);
	else {
		if (incx < 0) dx += (-n + 1) * incx;
		if (incy < 0) dy += (-n + 1) * incy;
		for (i = 0; i < n; i++) {
			dot_product += (*dx) * (*dy);
			dx += incx;
			dy += incy;
		}
	}
	return(dot_product);
}

/************************************************************** 
 * Constant times a vector plus a vector     		      *
 * Based on Fortran-77 routine from Linpack by J. Dongarra    *
 **************************************************************/ 
void svd_daxpy(long n, double da, double *dx, long incx, double *dy, long incy) {
	long i;

	if (n <= 0 || incx == 0 || incy == 0 || da == 0.0) return;
	if (incx == 1 && incy == 1)
		for (i = 0; i < n; i++) {
			*dy += da * (*dx++);
			dy++;
		}
	else {
		if (incx < 0) dx += (-n + 1) * incx;
		if (incy < 0) dy += (-n + 1) * incy;
		for (i = 0; i < n; i++) {
			*dy += da * (*dx);
			dx += incx;
			dy += incy;
		}
	}
	return;
}

/********************************************************************* 
 * Function sorts array1 and array2 into increasing order for array1 *
 *********************************************************************/
void svd_dsort2(long igap, long n, double *array1, double *array2) {
	double temp;
	long i, j, index;

	if (!igap) return;
	else {
		for (i = igap; i < n; i++) {
			j = i - igap;
			index = i;
			while (j >= 0 && array1[j] > array1[index]) {
				temp = array1[j];
				array1[j] = array1[index];
				array1[index] = temp;
				temp = array2[j];
				array2[j] = array2[index];
				array2[index] = temp;
				j -= igap;
				index = j + igap;
			}
		}
	}
	svd_dsort2(igap / 2, n, array1, array2);
}

/************************************************************** 
 * Function interchanges two vectors		     	      *
 * Based on Fortran-77 routine from Linpack by J. Dongarra    *
 **************************************************************/ 
void svd_dswap(long n, double *dx, long incx, double *dy, long incy) {
	long i;
	double dtemp;

	if (n <= 0 || incx == 0 || incy == 0) return;
	if (incx == 1 && incy == 1) {
		for (i = 0; i < n; i++) {
			dtemp = *dy;
			*dy++ = *dx;
			*dx++ = dtemp;
		}
	}
	else {
		if (incx < 0) dx += (-n + 1) * incx;
		if (incy < 0) dy += (-n + 1) * incy;
		for (i = 0; i < n; i++) {
			dtemp = *dy;
			*dy = *dx;
			*dx = dtemp;
			dx += incx;
			dy += incy;
		}
	}
}

/***************************************************************** 
 * Function finds the index of element having max. absolute value*
 * based on FORTRAN 77 routine from Linpack by J. Dongarra       *
 *****************************************************************/ 
long svd_idamax(long n, double *dx, long incx) {
	long ix, i, imax;
	double dtemp, dmax;

	if (n < 1) return(-1);
	if (n == 1) return(0);
	if (incx == 0) return(-1);

	if (incx < 0) ix = (-n + 1) * incx;
	else ix = 0;
	imax = ix;
	dx += ix;
	dmax = fabs(*dx);
	for (i = 1; i < n; i++) {
		ix += incx;
		dx += incx;
		dtemp = fabs(*dx);
		if (dtemp > dmax) {
			dmax = dtemp;
			imax = ix;
		}
	}
	return(imax);
}

/**************************************************************
 * multiplication of matrix B by vector x, where B = A'A,     *
 * and A is nrow by ncol (nrow >> ncol). Hence, B is of order *
 * n = ncol (y stores product vector).		              *
 **************************************************************/
void svd_opb_old(SMat A, double *x, double *y, double *temp) {
	long i, j, end;
	long *pointr = A->pointr, *rowind = A->rowind;
	double *value = A->value;
	long n = A->cols;

	SVDCount[SVD_MXV] += 2;
	memset(y, 0, n * sizeof(double));
	for (i = 0; i < A->rows; i++) temp[i] = 0.0;

	for (i = 0; i < A->cols; i++) {
		end = pointr[i + 1];
		for (j = pointr[i]; j < end; j++)
			temp[rowind[j]] += value[j] * (*x);
		x++;
	}
	for (i = 0; i < A->cols; i++) {
		end = pointr[i + 1];
		for (j = pointr[i]; j < end; j++)
			*y += value[j] * temp[rowind[j]];
		y++;
	}
	return;
}

// The following OpenMP version of the function only works when A is symmetric.
void svd_opb(SMat A, double *x, double *y, double *temp) {
	long i, j, end;
	long *pointr = A->pointr, *rowind = A->rowind;
	double *value = A->value;
	long n = A->cols;

	SVDCount[SVD_MXV] += 2;

#pragma omp parallel default(none) private(i, j, end) shared(n, pointr, rowind, value, x, temp)
	{
#pragma omp for
		for (i = 0; i < n; i++) {
			temp[i] = 0.0;
			end = pointr[i + 1];
			for (j = pointr[i]; j < end; j++)
				temp[i] += value[j] * x[rowind[j]];
		}
	}

#pragma omp parallel default(none) private(i, j, end) shared(n, pointr, rowind, value, y, temp)
	{
#pragma omp for
		for (i = 0; i < n; i++) {
			y[i] = 0.0;
			end = pointr[i + 1];
			for (j = pointr[i]; j < end; j++)
				y[i] += value[j] * temp[rowind[j]];
		}
	}
	return;
}


/***********************************************************
 * multiplication of matrix A by vector x, where A is 	   *
 * nrow by ncol (nrow >> ncol).  y stores product vector.  *
 ***********************************************************/
void svd_opa_old(SMat A, double *x, double *y) {
	long end, i, j;
	long *pointr = A->pointr, *rowind = A->rowind;
	double *value = A->value;

	SVDCount[SVD_MXV]++;
	memset(y, 0, A->rows * sizeof(double));

	for (i = 0; i < A->cols; i++) {
		end = pointr[i + 1];
		for (j = pointr[i]; j < end; j++)
			y[rowind[j]] += value[j] * x[i];
	}
	return;
}

// The following OpenMP version of the function only works when A is symmetric.
void svd_opa(SMat A, double *x, double *y) {
	long end, i, j;
	long *pointr = A->pointr, *rowind = A->rowind;
	double *value = A->value;
	long n = A->cols;

	SVDCount[SVD_MXV]++;

#pragma omp parallel default(none) private(i, j, end) shared(n, pointr, rowind, value, x, y)
	{
#pragma omp for
		for (i = 0; i < n; i++) {
			y[i] = 0.0;
			end = pointr[i + 1];
			for (j = pointr[i]; j < end; j++)
				y[i] += value[j] * x[rowind[j]];
		}
	}
	return;
}


/***********************************************************************
 *                                                                     *
 *				random()                               *
 *                        (double precision)                           *
 ***********************************************************************/
/***********************************************************************

   Description
   -----------

   This is a translation of a Fortran-77 uniform random number
   generator.  The code is based  on  theory and suggestions  given in
   D. E. Knuth (1969),  vol  2.  The argument to the function should 
   be initialized to an arbitrary integer prior to the first call to 
   random.  The calling program should  not  alter  the value of the
   argument between subsequent calls to random.  Random returns values
   within the interval (0,1).


   Arguments 
   ---------

   (input)
   iy	   an integer seed whose value must not be altered by the caller
	   between subsequent calls

   (output)
   random  a double precision random number between (0,1)

 ***********************************************************************/
double svd_random2(long *iy) {
	static long m2 = 0;
	static long ia, ic, mic;
	static double halfm, s;

	/* If first entry, compute (max int) / 2 */
	if (!m2) {
		m2 = 1 << (8 * (int)sizeof(int) - 2);
		halfm = m2;

		/* compute multiplier and increment for linear congruential
		 * method */
		ia = 8 * (long)(halfm * atan(1.0) / 8.0) + 5;
		ic = 2 * (long)(halfm * (0.5 - sqrt(3.0) / 6.0)) + 1;
		mic = (m2 - ic) + m2;

		/* s is the scale factor for converting to floating point */
		s = 0.5 / halfm;
	}

	/* compute next random number */
	*iy = *iy * ia;

	/* for computers which do not allow integer overflow on addition */
	if (*iy > mic) *iy = (*iy - m2) - m2;

	*iy = *iy + ic;

	/* for computers whose word length for addition is greater than
	 * for multiplication */
	if (*iy / 2 > m2) *iy = (*iy - m2) - m2;

	/* for computers whose integer overflow affects the sign bit */
	if (*iy < 0) *iy = (*iy + m2) + m2;

	return((double)(*iy) * s);
}

/************************************************************** 
 *							      *
 * Function finds sqrt(a^2 + b^2) without overflow or         *
 * destructive underflow.				      *
 *							      *
 **************************************************************/ 
/************************************************************** 

   Funtions used
   -------------

   UTILITY	dmax, dmin

 **************************************************************/ 
double svd_pythag(double a, double b) {
	double p, r, s, t, u, temp;

	p = svd_dmax(fabs(a), fabs(b));
	if (p != 0.0) {
		temp = svd_dmin(fabs(a), fabs(b)) / p;
		r = temp * temp;
		t = 4.0 + r;
		while (t != 4.0) {
			s = r / t;
			u = 1.0 + 2.0 * s;
			p *= u;
			temp = s / u;
			r *= temp * temp;
			t = 4.0 + r;
		}
	}
	return(p);
}


void svdResetCounters(void) {
	int i;
	for (i = 0; i < SVD_COUNTERS; i++)
		SVDCount[i] = 0;
}

/********************************* Allocation ********************************/

/* Row major order.  Rows are vectors that are consecutive in memory.  Matrix
   is initialized to empty. */
DMat svdNewDMat(int rows, int cols) {
	int i;
	DMat D = (DMat)malloc(sizeof(struct dmat));
	if (!D) { perror("svdNewDMat"); return NULL; }
	D->rows = rows;
	D->cols = cols;

	D->value = (double **)malloc(rows * sizeof(double *));
	if (!D->value) { SAFE_FREE(D); return NULL; }

	D->value[0] = (double *)calloc(rows * cols, sizeof(double));
	if (!D->value[0]) { SAFE_FREE(D->value); SAFE_FREE(D); return NULL; }

	for (i = 1; i < rows; i++) D->value[i] = D->value[i - 1] + cols;
	return D;
}

void svdFreeDMat(DMat D) {
	if (!D) return;
	SAFE_FREE(D->value[0]);
	SAFE_FREE(D->value);
	free(D);
}


SMat svdNewSMat(int rows, int cols, int vals) {
	SMat S = (SMat)calloc(1, sizeof(struct smat));
	if (!S) { perror("svdNewSMat"); return NULL; }
	S->rows = rows;
	S->cols = cols;
	S->vals = vals;
	S->pointr = svd_longArray(cols + 1, TRUE, "svdNewSMat: pointr");
	if (!S->pointr) { svdFreeSMat(S); return NULL; }
	S->rowind = svd_longArray(vals, FALSE, "svdNewSMat: rowind");
	if (!S->rowind) { svdFreeSMat(S); return NULL; }
	S->value = svd_doubleArray(vals, FALSE, "svdNewSMat: value");
	if (!S->value) { svdFreeSMat(S); return NULL; }
	return S;
}

void svdFreeSMat(SMat S) {
	if (!S) return;
	SAFE_FREE(S->pointr);
	SAFE_FREE(S->rowind);
	SAFE_FREE(S->value);
	free(S);
}


/* Creates an empty SVD record */
SVDRec svdNewSVDRec(void) {
	SVDRec R = (SVDRec)calloc(1, sizeof(struct svdrec));
	if (!R) { perror("svdNewSVDRec"); return NULL; }
	return R;
}

/* Frees an svd rec and all its contents. */
void svdFreeSVDRec(SVDRec R) {
	if (!R) return;
	if (R->Ut) svdFreeDMat(R->Ut);
	if (R->S) SAFE_FREE(R->S);
	if (R->Vt) svdFreeDMat(R->Vt);
	free(R);
}


/**************************** Conversion *************************************/

/* Efficiently transposes a sparse matrix. */
SMat svdTransposeS(SMat S) {
	int r, c, i, j;
	SMat N = svdNewSMat(S->cols, S->rows, S->vals);
	/* Count number nz in each row. */
	for (i = 0; i < S->vals; i++)
		N->pointr[S->rowind[i]]++;
	/* Fill each cell with the starting point of the previous row. */
	N->pointr[S->rows] = S->vals - N->pointr[S->rows - 1];
	for (r = S->rows - 1; r > 0; r--)
		N->pointr[r] = N->pointr[r + 1] - N->pointr[r - 1];
	N->pointr[0] = 0;
	/* Assign the new columns and values. */
	for (c = 0, i = 0; c < S->cols; c++) {
		for (; i < S->pointr[c + 1]; i++) {
			r = S->rowind[i];
			j = N->pointr[r + 1]++;
			N->rowind[j] = c;
			N->value[j] = S->value[i];
		}
	}
	return N;
}


/**************************** Input/Output ***********************************/

void svdWriteDenseArray(double *a, int n, char *filename) {
	int i;
	FILE *file = svd_writeFile(filename, FALSE);
	if (!file)
		return svd_error("svdWriteDenseArray: failed to write %s", filename);
	fprintf(file, "%d\n", n);
	for (i = 0; i < n; i++)
		fprintf(file, "%g\n", a[i]);
	svd_closeFile(file);
}

void writeSparseBinaryFile(SMat S, char *filename) {
	FILE *file = fopen(filename, "wb");
	if (file == NULL)
		return;

	int c, v;
	for (c = 0, v = 0; c < S->cols; c++) {
		for (; v < S->pointr[c + 1]; v++) {
			fwrite(&c, sizeof(int), 1, file);
			fwrite(&S->rowind[v], sizeof(int), 1, file);
			fwrite(&S->value[v], sizeof(double), 1, file);
		}
	}
	fclose(file);
}

static void svdWriteDenseTextFile(DMat D, FILE *file) {
	int i, j;
	fprintf(file, "%ld %ld\n", D->rows, D->cols);
	for (i = 0; i < D->rows; i++)
		for (j = 0; j < D->cols; j++)
			fprintf(file, "%g%c", D->value[i][j], (j == D->cols - 1) ? '\n' : ' ');
}


void svdWriteDenseMatrix(DMat D, char *filename) {
	SMat S = NULL;
	FILE *file = svd_writeFile(filename, FALSE);
	if (!file) {
		svd_error("svdWriteDenseMatrix: failed to write file %s\n", filename);
		return;
	}
	svdWriteDenseTextFile(D, file);
	svd_closeFile(file);
	if (S) svdFreeSMat(S);
}


/***********************************************************************
 *                                                                     *
 *                        main()                                       *
 * Sparse SVD(A) via Eigensystem of A'A symmetric Matrix 	       *
 *                  (double precision)                                 *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

   Description
   -----------

   This sample program uses landr to compute singular triplets of A via
   the equivalent symmetric eigenvalue problem                         

   B x = lambda x, where x' = (u',v'), lambda = sigma**2,
   where sigma is a singular value of A,
                                                                     
   B = A'A , and A is m (nrow) by n (ncol) (nrow >> ncol),                
                                                                 
   so that {u,sqrt(lambda),v} is a singular triplet of A.        
   (A' = transpose of A)                                      
                                                            
   User supplied routines: svd_opa, opb, store, timer              
                                                        
   svd_opa(     x,y) takes an n-vector x and returns A*x in y.
   svd_opb(ncol,x,y) takes an n-vector x and returns B*x in y.
                                                                  
   Based on operation flag isw, store(n,isw,j,s) stores/retrieves 
   to/from storage a vector of length n in s.                   
                                                               
   User should edit timer() with an appropriate call to an intrinsic
   timing routine that returns elapsed user time.                      


   External parameters 
   -------------------

   Defined and documented in las2.h


   Local parameters 
   ----------------

  (input)
   endl     left end of interval containing unwanted eigenvalues of B
   endr     right end of interval containing unwanted eigenvalues of B
   kappa    relative accuracy of ritz values acceptable as eigenvalues
              of B
	      vectors is not equal to 1
   r        work array
   n	    dimension of the eigenproblem for matrix B (ncol)
   dimensions   upper limit of desired number of singular triplets of A
   iterations   upper limit of desired number of Lanczos steps
   nnzero   number of nonzeros in A
   vectors  1 indicates both singular values and singular vectors are 
	      wanted and they can be found in output file lav2;
	      0 indicates only singular values are wanted 
   		
  (output)
   ritz	    array of ritz values
   bnd      array of error bounds
   d        array of singular values
   memory   total memory allocated in bytes to solve the B-eigenproblem


   Functions used
   --------------

   BLAS		svd_daxpy, svd_dscal, svd_ddot
   USER		svd_opa, svd_opb, timer
   MISC		write_header, check_parameters
   LAS2		landr


   Precision
   ---------

   All floating-point calculations are done in double precision;
   variables are declared as long and double.


   LAS2 development
   ----------------

   LAS2 is a C translation of the Fortran-77 LAS2 from the SVDPACK
   library written by Michael W. Berry, University of Tennessee,
   Dept. of Computer Science, 107 Ayres Hall, Knoxville, TN, 37996-1301

   31 Jan 1992:  Date written 

   Theresa H. Do
   University of Tennessee
   Dept. of Computer Science
   107 Ayres Hall
   Knoxville, TN, 37996-1301
   internet: tdo@cs.utk.edu

 ***********************************************************************/

/***********************************************************************
 *								       *
 *		      check_parameters()			       *
 *								       *
 ***********************************************************************/
/***********************************************************************

   Description
   -----------
   Function validates input parameters and returns error code (long)  

   Parameters 
   ----------
  (input)
   dimensions   upper limit of desired number of eigenpairs of B           
   iterations   upper limit of desired number of lanczos steps             
   n        dimension of the eigenproblem for matrix B               
   endl     left end of interval containing unwanted eigenvalues of B
   endr     right end of interval containing unwanted eigenvalues of B
   vectors  1 indicates both eigenvalues and eigenvectors are wanted 
            and they can be found in lav2; 0 indicates eigenvalues only
   nnzero   number of nonzero elements in input matrix (matrix A)      
                                                                      
 ***********************************************************************/

long check_parameters(SMat A, long dimensions, long iterations,
	double endl, double endr, long vectors) {
	long error_index;
	error_index = 0;

	if (endl >/*=*/ endr)  error_index = 2;
	else if (dimensions > iterations) error_index = 3;
	else if (A->cols <= 0 || A->rows <= 0) error_index = 4;
	/*else if (n > A->cols || n > A->rows) error_index = 1;*/
	else if (iterations <= 0 || iterations > A->cols || iterations > A->rows)
		error_index = 5;
	else if (dimensions <= 0 || dimensions > iterations) error_index = 6;
	if (error_index)
		svd_error("svdLAS2 parameter error: %s\n", error_msg[error_index]);
	return(error_index);
}

/***********************************************************************
 *								       *
 *			  write_header()			       *
 *   Function writes out header of output file containing ritz values  *
 *								       *
 ***********************************************************************/

void write_header(long iterations, long dimensions, double endl, double endr,
	long vectors, double kappa, long nrow, long ncol,
	long vals) {
	printf("NO. OF ROWS               = %6ld\n", nrow);
	printf("NO. OF COLUMNS            = %6ld\n", ncol);
	printf("NO. OF NON-ZERO VALUES    = %6ld\n", vals);
	printf("MATRIX DENSITY            = %6.2f%%\n", ((float)vals / nrow) * 100 / ncol);
	printf("\n");
	return;
}


/***********************************************************************
 *                                                                     *
 *				landr()				       *
 *        Lanczos algorithm with selective orthogonalization           *
 *                    Using Simon's Recurrence                         *
 *                       (double precision)                            *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

   Description
   -----------

   landr() is the LAS2 driver routine that, upon entry,
     (1)  checks for the validity of input parameters of the 
	  B-eigenproblem 
     (2)  determines several machine constants
     (3)  makes a Lanczos run
     (4)  calculates B-eigenvectors (singular vectors of A) if requested 
	  by user


   arguments
   ---------

   (input)
   n        dimension of the eigenproblem for A'A
   iterations   upper limit of desired number of Lanczos steps
   dimensions   upper limit of desired number of eigenpairs
   nnzero   number of nonzeros in matrix A
   endl     left end of interval containing unwanted eigenvalues of B
   endr     right end of interval containing unwanted eigenvalues of B
   vectors  1 indicates both eigenvalues and eigenvectors are wanted
              and they can be found in output file lav2; 
	    0 indicates only eigenvalues are wanted
   kappa    relative accuracy of ritz values acceptable as eigenvalues
	      of B (singular values of A)
   r        work array

   (output)
   j        number of Lanczos steps actually taken                     
   neig     number of ritz values stabilized                           
   ritz     array to hold the ritz values                              
   bnd      array to hold the error bounds


   External parameters
   -------------------

   Defined and documented in las2.h


   local parameters
   -------------------

   ibeta    radix for the floating-point representation
   it       number of base ibeta digits in the floating-point significand
   irnd     floating-point addition rounded or chopped
   machep   machine relative precision or round-off error
   negeps   largest negative integer
   wptr	    array of pointers each pointing to a work space


   Functions used
   --------------

   MISC         svd_dmax, machar, check_parameters
   LAS2         ritvec, lanso

 ***********************************************************************/

SVDRec svdLAS2A(SMat A, long dimensions) {
	double end[2] = { -1.0e-30, 1.0e-30 };
	double kappa = 1e-6;
	if (!A) {
		svd_error("svdLAS2A called with NULL array\n");
		return NULL;
	}
	return svdLAS2(A, dimensions, 0, end, kappa);
}


SVDRec svdLAS2(SMat A, long dimensions, long iterations, double end[2],
	double kappa) {
	char transpose = FALSE;
	long ibeta, it, irnd, machep, negep, n, i, steps, nsig, neig, m;
	double *wptr[10], *ritz, *bnd;
	SVDRec R = NULL;
	ierr = 0;  // reset the global error flag

	svdResetCounters();

	m = svd_imin(A->rows, A->cols);
	if (dimensions <= 0 || dimensions > m)
		dimensions = m;
	if (iterations <= 0 || iterations > m)
		iterations = m;
	if (iterations < dimensions) iterations = dimensions;

	/* Write output header */
	if (SVDVerbosity > 0)
		write_header(iterations, dimensions, end[0], end[1], TRUE, kappa, A->rows, A->cols, A->vals);

	/* Check parameters */
	if (check_parameters(A, dimensions, iterations, end[0], end[1], TRUE))
		return NULL;

	/* If A is wide, the SVD is computed on its transpose for speed. */
	if (A->cols >= A->rows * 1.2) {
		if (SVDVerbosity > 0) printf("TRANSPOSING THE MATRIX FOR SPEED\n");
		transpose = TRUE;
		A = svdTransposeS(A);
	}

	n = A->cols;
	/* Compute machine precision */
	machar(&ibeta, &it, &irnd, &machep, &negep);
	eps1 = eps * sqrt((double)n);
	reps = sqrt(eps);
	eps34 = reps * sqrt(reps);

	/* Allocate temporary space. */
	if (!(wptr[0] = svd_doubleArray(n, TRUE, "las2: wptr[0]"))) goto abort;
	if (!(wptr[1] = svd_doubleArray(n, FALSE, "las2: wptr[1]"))) goto abort;
	if (!(wptr[2] = svd_doubleArray(n, FALSE, "las2: wptr[2]"))) goto abort;
	if (!(wptr[3] = svd_doubleArray(n, FALSE, "las2: wptr[3]"))) goto abort;
	if (!(wptr[4] = svd_doubleArray(n, FALSE, "las2: wptr[4]"))) goto abort;
	if (!(wptr[5] = svd_doubleArray(n, FALSE, "las2: wptr[5]"))) goto abort;
	if (!(wptr[6] = svd_doubleArray(iterations, FALSE, "las2: wptr[6]")))
		goto abort;
	if (!(wptr[7] = svd_doubleArray(iterations, FALSE, "las2: wptr[7]")))
		goto abort;
	if (!(wptr[8] = svd_doubleArray(iterations, FALSE, "las2: wptr[8]")))
		goto abort;
	if (!(wptr[9] = svd_doubleArray(iterations + 1, FALSE, "las2: wptr[9]")))
		goto abort;
	/* Calloc may be unnecessary: */
	if (!(ritz = svd_doubleArray(iterations + 1, TRUE, "las2: ritz")))
		goto abort;
	/* Calloc may be unnecessary: */
	if (!(bnd = svd_doubleArray(iterations + 1, TRUE, "las2: bnd")))
		goto abort;
	memset(bnd, 127, (iterations + 1) * sizeof(double));

	if (!(LanStore = (double **)calloc(iterations + MAXLL, sizeof(double *))))
		goto abort;
	if (!(OPBTemp = svd_doubleArray(A->rows, FALSE, "las2: OPBTemp")))
		goto abort;

	/* Actually run the lanczos thing: */
	steps = lanso(A, iterations, dimensions, end[0], end[1], ritz, bnd, wptr,
		&neig, n);

	/* Print some stuff. */
	//if (SVDVerbosity > 0) {
	//	printf("NUMBER OF LANCZOS STEPS   = %6ld\n"
	//		"RITZ VALUES STABILIZED    = %6ld\n", steps + 1, neig);
	//}
	if (SVDVerbosity > 2) {
		printf("\nCOMPUTED RITZ VALUES  (ERROR BNDS)\n");
		for (i = 0; i <= steps; i++)
			printf("%3ld  %22.14E  (%11.2E)\n", i + 1, ritz[i], bnd[i]);
	}

	SAFE_FREE(wptr[0]);
	SAFE_FREE(wptr[1]);
	SAFE_FREE(wptr[2]);
	SAFE_FREE(wptr[3]);
	SAFE_FREE(wptr[4]);
	SAFE_FREE(wptr[7]);
	SAFE_FREE(wptr[8]);

	/* Compute eigenvectors */
	kappa = svd_dmax(fabs(kappa), eps34);

	R = svdNewSVDRec();
	if (!R) {
		svd_error("svdLAS2: allocation of R failed");
		goto cleanup;
	}
	R->d = /*svd_imin(nsig, dimensions)*/dimensions;
	R->Ut = svdNewDMat(R->d, A->rows);
	R->S = svd_doubleArray(R->d, TRUE, "las2: R->s");
	R->Vt = svdNewDMat(R->d, A->cols);
	if (!R->Ut || !R->S || !R->Vt) {
		svd_error("svdLAS2: allocation of R failed");
		goto cleanup;
	}

	nsig = ritvec(n, A, R, kappa, ritz, bnd, wptr[6], wptr[9], wptr[5], steps,
		neig);

	if (SVDVerbosity > 1) {
		printf("\nSINGULAR VALUES: ");
		svdWriteDenseArray(R->S, R->d, "-");

		if (SVDVerbosity > 2) {
			printf("\nLEFT SINGULAR VECTORS (transpose of U): ");
			svdWriteDenseMatrix(R->Ut, "-");

			printf("\nRIGHT SINGULAR VECTORS (transpose of V): ");
			svdWriteDenseMatrix(R->Vt, "-");
		}
	}
	if (SVDVerbosity > 0) {
		printf("SINGULAR VALUES FOUND     = %6d\n"
			"SIGNIFICANT VALUES        = %6ld\n", R->d, nsig);
	}

cleanup:
	for (i = 0; i <= 9; i++)
		SAFE_FREE(wptr[i]);
	SAFE_FREE(ritz);
	SAFE_FREE(bnd);
	if (LanStore) {
		for (i = 0; i < iterations + MAXLL; i++)
			SAFE_FREE(LanStore[i]);
		SAFE_FREE(LanStore);
	}
	SAFE_FREE(OPBTemp);

	/* This swaps and transposes the singular matrices if A was transposed. */
	if (R && transpose) {
		DMat T;
		svdFreeSMat(A);
		T = R->Ut;
		R->Ut = R->Vt;
		R->Vt = T;
	}

	return R;
abort:
	svd_error("svdLAS2: fatal error, aborting");
	return NULL;
}


/***********************************************************************
 *                                                                     *
 *                        ritvec()                                     *
 * 	    Function computes the singular vectors of matrix A	       *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

   Description
   -----------

   This function is invoked by landr() only if eigenvectors of the A'A
   eigenproblem are desired.  When called, ritvec() computes the 
   singular vectors of A and writes the result to an unformatted file.


   Parameters
   ----------

   (input)
   nrow       number of rows of A
   steps      number of Lanczos iterations performed
   fp_out2    pointer to unformatted output file
   n	      dimension of matrix A
   kappa      relative accuracy of ritz values acceptable as 
		eigenvalues of A'A
   ritz       array of ritz values
   bnd        array of error bounds
   alf        array of diagonal elements of the tridiagonal matrix T
   bet        array of off-diagonal elements of T
   w1, w2     work space

   (output)
   xv1        array of eigenvectors of A'A (right singular vectors of A)
   ierr	      error code
              0 for normal return from imtql2()
	      k if convergence did not occur for k-th eigenvalue in
	        imtql2()
   nsig       number of accepted ritz values based on kappa

   (local)
   s	      work array which is initialized to the identity matrix
	      of order (j + 1) upon calling imtql2().  After the call,
	      s contains the orthonormal eigenvectors of the symmetric 
	      tridiagonal matrix T

   Functions used
   --------------

   BLAS		svd_dscal, svd_dcopy, svd_daxpy
   USER		store
   		imtql2

 ***********************************************************************/

void rotateArray(double *a, int size, int x) {
	int i, j, n, start;
	double t1, t2;
	if (x == 0) return;
	j = start = 0;
	t1 = a[0];
	for (i = 0; i < size; i++) {
		n = (j >= x) ? j - x : j + size - x;
		t2 = a[n];
		a[n] = t1;
		t1 = t2;
		j = n;
		if (j == start) {
			start = ++j;
			t1 = a[j];
		}
	}
}

long ritvec(long n, SMat A, SVDRec R, double kappa, double *ritz, double *bnd,
	double *alf, double *bet, double *w2, long steps, long neig) {
	long js, jsq, i, k, /*size,*/ id2, tmp, nsig, x;
	double *s, *xv2, tmp0, tmp1, xnorm, *w1 = R->Vt->value[0];

	js = steps + 1;
	jsq = js * js;
	/*size = sizeof(double) * n;*/

	s = svd_doubleArray(jsq, TRUE, "ritvec: s");
	xv2 = svd_doubleArray(n, FALSE, "ritvec: xv2");

	/* initialize s to an identity matrix */
	for (i = 0; i < jsq; i += (js + 1)) s[i] = 1.0;
	svd_dcopy(js, alf, 1, w1, -1);
	svd_dcopy(steps, &bet[1], 1, &w2[1], -1);

	/* on return from imtql2(), w1 contains eigenvalues in ascending
	 * order and s contains the corresponding eigenvectors */
	imtql2(js, js, w1, w2, s);

	/*fwrite((char *)&n, sizeof(n), 1, fp_out2);
	  fwrite((char *)&js, sizeof(js), 1, fp_out2);
	  fwrite((char *)&kappa, sizeof(kappa), 1, fp_out2);*/
	  /*id = 0;*/
	nsig = 0;

	if (ierr) {
		R->d = 0;
	}
	else {
		x = 0;
		id2 = jsq - js;
		for (k = 0; k < js; k++) {
			tmp = id2;
			if (bnd[k] <= kappa * fabs(ritz[k]) && k > js - neig - 1) {
				if (--x < 0) x = R->d - 1;
				w1 = R->Vt->value[x];
				for (i = 0; i < n; i++) w1[i] = 0.0;
				for (i = 0; i < js; i++) {
					store(n, RETRQ, i, w2);
					svd_daxpy(n, s[tmp], w2, 1, w1, 1);
					tmp -= js;
				}
				/*fwrite((char *)w1, size, 1, fp_out2);*/

				/* store the w1 vector row-wise in array xv1;
				 * size of xv1 is (steps+1) * (nrow+ncol) elements
				 * and each vector, even though only ncol long,
				 * will have (nrow+ncol) elements in xv1.
				 * It is as if xv1 is a 2-d array (steps+1) by
				 * (nrow+ncol) and each vector occupies a row  */

				 /* j is the index in the R arrays, which are sorted by high to low
					singular values. */

					/*for (i = 0; i < n; i++) R->Vt->value[x]xv1[id++] = w1[i];*/
					/*id += nrow;*/
				nsig++;
			}
			id2++;
		}
		SAFE_FREE(s);

		/* Rotate the singular vectors and values. */
		/* x is now the location of the highest singular value. */
		rotateArray(R->Vt->value[0], R->Vt->rows * R->Vt->cols,
			x * R->Vt->cols);
		R->d = svd_imin(R->d, nsig);
		for (x = 0; x < R->d; x++) {
			/* multiply by matrix B first */
			svd_opb(A, R->Vt->value[x], xv2, OPBTemp);
			tmp0 = svd_ddot(n, R->Vt->value[x], 1, xv2, 1);
			svd_daxpy(n, -tmp0, R->Vt->value[x], 1, xv2, 1);
			tmp0 = sqrt(tmp0);
			xnorm = sqrt(svd_ddot(n, xv2, 1, xv2, 1));

			/* multiply by matrix A to get (scaled) left s-vector */
			svd_opa(A, R->Vt->value[x], R->Ut->value[x]);
			tmp1 = 1.0 / tmp0;
			svd_dscal(A->rows, tmp1, R->Ut->value[x], 1);
			xnorm *= tmp1;
			bnd[i] = xnorm;
			R->S[x] = tmp0;
		}
	}

	SAFE_FREE(s);
	SAFE_FREE(xv2);
	return nsig;
}

/***********************************************************************
 *                                                                     *
 *                          lanso()                                    *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

   Description
   -----------

   Function determines when the restart of the Lanczos algorithm should 
   occur and when it should terminate.

   Arguments 
   ---------

   (input)
   n         dimension of the eigenproblem for matrix B
   iterations    upper limit of desired number of lanczos steps           
   dimensions    upper limit of desired number of eigenpairs             
   endl      left end of interval containing unwanted eigenvalues
   endr      right end of interval containing unwanted eigenvalues
   ritz      array to hold the ritz values                       
   bnd       array to hold the error bounds                          
   wptr      array of pointers that point to work space:            
  	       wptr[0]-wptr[5]  six vectors of length n		
  	       wptr[6] array to hold diagonal of the tridiagonal matrix T
  	       wptr[9] array to hold off-diagonal of T	
  	       wptr[7] orthogonality estimate of Lanczos vectors at 
		 step j
 	       wptr[8] orthogonality estimate of Lanczos vectors at 
		 step j-1

   (output)
   j         number of Lanczos steps actually taken
   neig      number of ritz values stabilized
   ritz      array to hold the ritz values
   bnd       array to hold the error bounds
   ierr      (globally declared) error flag
	     ierr = 8192 if stpone() fails to find a starting vector
	     ierr = k if convergence did not occur for k-th eigenvalue
		    in imtqlb()
	     ierr = 0 otherwise


   Functions used
   --------------

   LAS		stpone, error_bound, lanczos_step
   MISC		svd_dsort2
   UTILITY	svd_imin, svd_imax

 ***********************************************************************/

int lanso(SMat A, long iterations, long dimensions, double endl,
	double endr, double *ritz, double *bnd, double *wptr[],
	long *neigp, long n) {
	double *alf, *eta, *oldeta, *bet, *wrk, rnm, tol;
	long ll, first, last, ENOUGH, id2, id3, i, l, neig, j = 0, intro = 0;

	alf = wptr[6];
	eta = wptr[7];
	oldeta = wptr[8];
	bet = wptr[9];
	wrk = wptr[5];

	/* take the first step */
	stpone(A, wptr, &rnm, &tol, n);
	if (!rnm || ierr) return 0;
	eta[0] = eps1;
	oldeta[0] = eps1;
	ll = 0;
	first = 1;
	last = svd_imin(dimensions + svd_imax(8, dimensions), iterations);
	ENOUGH = FALSE;
	/*id1 = 0;*/
	while (/*id1 < dimensions && */!ENOUGH) {
		if (rnm <= tol) rnm = 0.0;

		/* the actual lanczos loop */
		j = lanczos_step(A, first, last, wptr, alf, eta, oldeta, bet, &ll,
			&ENOUGH, &rnm, &tol, n);
		if (ENOUGH) j = j - 1;
		else j = last - 1;
		first = j + 1;
		bet[j + 1] = rnm;

		/* analyze T */
		l = 0;
		for (id2 = 0; id2 < j; id2++) {
			if (l > j) break;
			for (i = l; i <= j; i++) if (!bet[i + 1]) break;
			if (i > j) i = j;

			/* now i is at the end of an unreduced submatrix */
			svd_dcopy(i - l + 1, &alf[l], 1, &ritz[l], -1);
			svd_dcopy(i - l, &bet[l + 1], 1, &wrk[l + 1], -1);

			imtqlb(i - l + 1, &ritz[l], &wrk[l], &bnd[l]);

			if (ierr) {
				svd_error("svdLAS2: imtqlb failed to converge (ierr = %ld)\n", ierr);
				svd_error("  l = %ld  i = %ld\n", l, i);
				for (id3 = l; id3 <= i; id3++)
					svd_error("  %ld  %lg  %lg  %lg\n",
						id3, ritz[id3], wrk[id3], bnd[id3]);
			}
			for (id3 = l; id3 <= i; id3++)
				bnd[id3] = rnm * fabs(bnd[id3]);
			l = i + 1;
		}

		/* sort eigenvalues into increasing order */
		svd_dsort2((j + 1) / 2, j + 1, ritz, bnd);

		/*    for (i = 0; i < iterations; i++)
		  printf("%f ", ritz[i]);
		  printf("\n"); */

		  /* massage error bounds for very close ritz values */
		neig = error_bound(&ENOUGH, endl, endr, ritz, bnd, j, tol);
		*neigp = neig;

		/* should we stop? */
		if (neig < dimensions) {
			if (!neig) {
				last = first + 9;
				intro = first;
			}
			else last = first + svd_imax(3, 1 + ((j - intro) * (dimensions - neig)) /
				neig);
			last = svd_imin(last, iterations);
		}
		else ENOUGH = TRUE;
		ENOUGH = ENOUGH || first >= iterations;
		/* id1++; */
		/* printf("id1=%d dimen=%d first=%d\n", id1, dimensions, first); */
	}
	store(n, STORQ, j, wptr[1]);
	return j;
}


/***********************************************************************
 *                                                                     *
 *			lanczos_step()                                 *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

   Description
   -----------

   Function embodies a single Lanczos step

   Arguments 
   ---------

   (input)
   n        dimension of the eigenproblem for matrix B
   first    start of index through loop				      
   last     end of index through loop				     
   wptr	    array of pointers pointing to work space		    
   alf	    array to hold diagonal of the tridiagonal matrix T
   eta      orthogonality estimate of Lanczos vectors at step j   
   oldeta   orthogonality estimate of Lanczos vectors at step j-1
   bet      array to hold off-diagonal of T                     
   ll       number of intitial Lanczos vectors in local orthog. 
              (has value of 0, 1 or 2)			
   enough   stop flag			

   Functions used
   --------------

   BLAS		svd_ddot, svd_dscal, svd_daxpy, svd_datx, svd_dcopy
   USER		store
   LAS		purge, ortbnd, startv
   UTILITY	svd_imin, svd_imax

 ***********************************************************************/

long lanczos_step(SMat A, long first, long last, double *wptr[],
	double *alf, double *eta, double *oldeta,
	double *bet, long *ll, long *enough, double *rnmp,
	double *tolp, long n) {
	double t, *mid, rnm = *rnmp, tol = *tolp, anorm;
	long i, j;

	for (j = first; j < last; j++) {
		mid = wptr[2];
		wptr[2] = wptr[1];
		wptr[1] = mid;
		mid = wptr[3];
		wptr[3] = wptr[4];
		wptr[4] = mid;

		store(n, STORQ, j - 1, wptr[2]);
		if (j - 1 < MAXLL) store(n, STORP, j - 1, wptr[4]);
		bet[j] = rnm;

		/* restart if invariant subspace is found */
		if (!bet[j]) {
			rnm = startv(A, wptr, j, n);
			if (ierr) return j;
			if (!rnm) *enough = TRUE;
		}
		if (*enough) {
			/* added by Doug... */
			/* These lines fix a bug that occurs with low-rank matrices */
			mid = wptr[2];
			wptr[2] = wptr[1];
			wptr[1] = mid;
			/* ...added by Doug */
			break;
		}

		/* take a lanczos step */
		t = 1.0 / rnm;
		svd_datx(n, t, wptr[0], 1, wptr[1], 1);
		svd_dscal(n, t, wptr[3], 1);
		svd_opb(A, wptr[3], wptr[0], OPBTemp);
		svd_daxpy(n, -rnm, wptr[2], 1, wptr[0], 1);
		alf[j] = svd_ddot(n, wptr[0], 1, wptr[3], 1);
		svd_daxpy(n, -alf[j], wptr[1], 1, wptr[0], 1);

		/* orthogonalize against initial lanczos vectors */
		if (j <= MAXLL && (fabs(alf[j - 1]) > 4.0 * fabs(alf[j])))
			*ll = j;
		for (i = 0; i < svd_imin(*ll, j - 1); i++) {
			store(n, RETRP, i, wptr[5]);
			t = svd_ddot(n, wptr[5], 1, wptr[0], 1);
			store(n, RETRQ, i, wptr[5]);
			svd_daxpy(n, -t, wptr[5], 1, wptr[0], 1);
			eta[i] = eps1;
			oldeta[i] = eps1;
		}

		/* extended local reorthogonalization */
		t = svd_ddot(n, wptr[0], 1, wptr[4], 1);
		svd_daxpy(n, -t, wptr[2], 1, wptr[0], 1);
		if (bet[j] > 0.0) bet[j] = bet[j] + t;
		t = svd_ddot(n, wptr[0], 1, wptr[3], 1);
		svd_daxpy(n, -t, wptr[1], 1, wptr[0], 1);
		alf[j] = alf[j] + t;
		svd_dcopy(n, wptr[0], 1, wptr[4], 1);
		rnm = sqrt(svd_ddot(n, wptr[0], 1, wptr[4], 1));
		anorm = bet[j] + fabs(alf[j]) + rnm;
		tol = reps * anorm;

		/* update the orthogonality bounds */
		ortbnd(alf, eta, oldeta, bet, j, rnm);

		/* restore the orthogonality state when needed */
		purge(n, *ll, wptr[0], wptr[1], wptr[4], wptr[3], wptr[5], eta, oldeta,
			j, &rnm, tol);
		if (rnm <= tol) rnm = 0.0;
	}
	*rnmp = rnm;
	*tolp = tol;
	return j;
}

/***********************************************************************
 *                                                                     *
 *                          ortbnd()                                   *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

   Description
   -----------

   Funtion updates the eta recurrence

   Arguments 
   ---------

   (input)
   alf      array to hold diagonal of the tridiagonal matrix T         
   eta      orthogonality estimate of Lanczos vectors at step j        
   oldeta   orthogonality estimate of Lanczos vectors at step j-1     
   bet      array to hold off-diagonal of T                          
   n        dimension of the eigenproblem for matrix B		    
   j        dimension of T					  
   rnm	    norm of the next residual vector			 
   eps1	    roundoff estimate for dot product of two unit vectors

   (output)
   eta      orthogonality estimate of Lanczos vectors at step j+1     
   oldeta   orthogonality estimate of Lanczos vectors at step j        


   Functions used
   --------------

   BLAS		svd_dswap

 ***********************************************************************/

void ortbnd(double *alf, double *eta, double *oldeta, double *bet, long step,
			double rnm) {
	long i;
	if (step < 1) return;
	if (rnm) {
		if (step > 1) {
			oldeta[0] = (bet[1] * eta[1] + (alf[0] - alf[step]) * eta[0] -
				bet[step] * oldeta[0]) / rnm + eps1;
		}
		for (i = 1; i <= step - 2; i++)
			oldeta[i] = (bet[i + 1] * eta[i + 1] + (alf[i] - alf[step]) * eta[i] +
				bet[i] * eta[i - 1] - bet[step] * oldeta[i]) / rnm + eps1;
	}
	oldeta[step - 1] = eps1;
	svd_dswap(step, oldeta, 1, eta, 1);
	eta[step] = eps1;
	return;
}

/***********************************************************************
 *                                                                     *
 *				purge()                                *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

   Description
   -----------

   Function examines the state of orthogonality between the new Lanczos
   vector and the previous ones to decide whether re-orthogonalization 
   should be performed


   Arguments 
   ---------

   (input)
   n        dimension of the eigenproblem for matrix B		       
   ll       number of intitial Lanczos vectors in local orthog.       
   r        residual vector to become next Lanczos vector            
   q        current Lanczos vector			           
   ra       previous Lanczos vector
   qa       previous Lanczos vector
   wrk      temporary vector to hold the previous Lanczos vector
   eta      state of orthogonality between r and prev. Lanczos vectors 
   oldeta   state of orthogonality between q and prev. Lanczos vectors
   j        current Lanczos step				     

   (output)
   r	    residual vector orthogonalized against previous Lanczos 
	      vectors
   q        current Lanczos vector orthogonalized against previous ones


   Functions used
   --------------

   BLAS		svd_daxpy,  svd_dcopy,  svd_idamax,  svd_ddot
   USER		store

 ***********************************************************************/

void purge(long n, long ll, double *r, double *q, double *ra,
	double *qa, double *wrk, double *eta, double *oldeta, long step,
	double *rnmp, double tol) {
	double t, tq, tr, reps1, rnm = *rnmp;
	long k, iteration, flag, i;

	if (step < ll + 2) return;

	k = svd_idamax(step - (ll + 1), &eta[ll], 1) + ll;
	if (fabs(eta[k]) > reps) {
		reps1 = eps1 / reps;
		iteration = 0;
		flag = TRUE;
		while (iteration < 2 && flag) {
			if (rnm > tol) {

				/* bring in a lanczos vector t and orthogonalize both
				 * r and q against it */
				tq = 0.0;
				tr = 0.0;
				for (i = ll; i < step; i++) {
					store(n, RETRQ, i, wrk);
					t = -svd_ddot(n, qa, 1, wrk, 1);
					tq += fabs(t);
					svd_daxpy(n, t, wrk, 1, q, 1);
					t = -svd_ddot(n, ra, 1, wrk, 1);
					tr += fabs(t);
					svd_daxpy(n, t, wrk, 1, r, 1);
				}
				svd_dcopy(n, q, 1, qa, 1);
				t = -svd_ddot(n, r, 1, qa, 1);
				tr += fabs(t);
				svd_daxpy(n, t, q, 1, r, 1);
				svd_dcopy(n, r, 1, ra, 1);
				rnm = sqrt(svd_ddot(n, ra, 1, r, 1));
				if (tq <= reps1 && tr <= reps1 * rnm) flag = FALSE;
			}
			iteration++;
		}
		for (i = ll; i <= step; i++) {
			eta[i] = eps1;
			oldeta[i] = eps1;
		}
	}
	*rnmp = rnm;
	return;
}


/***********************************************************************
 *                                                                     *
 *                         stpone()                                    *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

   Description
   -----------

   Function performs the first step of the Lanczos algorithm.  It also
   does a step of extended local re-orthogonalization.

   Arguments 
   ---------

   (input)
   n      dimension of the eigenproblem for matrix B

   (output)
   ierr   error flag
   wptr   array of pointers that point to work space that contains
	    wptr[0]             r[j]
	    wptr[1]             q[j]
	    wptr[2]             q[j-1]
	    wptr[3]             p
	    wptr[4]             p[j-1]
	    wptr[6]             diagonal elements of matrix T 


   Functions used
   --------------

   BLAS		svd_daxpy, svd_datx, svd_dcopy, svd_ddot, svd_dscal
   USER		store, opb
   LAS		startv

 ***********************************************************************/

void stpone(SMat A, double *wrkptr[], double *rnmp, double *tolp, long n) {
	double t, *alf, rnm, anorm;
	alf = wrkptr[6];

	/* get initial vector; default is random */
	rnm = startv(A, wrkptr, 0, n);
	if (rnm == 0.0 || ierr != 0) return;

	/* normalize starting vector */
	t = 1.0 / rnm;
	svd_datx(n, t, wrkptr[0], 1, wrkptr[1], 1);
	svd_dscal(n, t, wrkptr[3], 1);

	/* take the first step */
	svd_opb(A, wrkptr[3], wrkptr[0], OPBTemp);
	alf[0] = svd_ddot(n, wrkptr[0], 1, wrkptr[3], 1);
	svd_daxpy(n, -alf[0], wrkptr[1], 1, wrkptr[0], 1);
	t = svd_ddot(n, wrkptr[0], 1, wrkptr[3], 1);
	svd_daxpy(n, -t, wrkptr[1], 1, wrkptr[0], 1);
	alf[0] += t;
	svd_dcopy(n, wrkptr[0], 1, wrkptr[4], 1);
	rnm = sqrt(svd_ddot(n, wrkptr[0], 1, wrkptr[4], 1));
	anorm = rnm + fabs(alf[0]);
	*rnmp = rnm;
	*tolp = reps * anorm;

	return;
}

/***********************************************************************
 *                                                                     *
 *                         startv()                                    *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

   Description
   -----------

   Function delivers a starting vector in r and returns |r|; it returns 
   zero if the range is spanned, and ierr is non-zero if no starting 
   vector within range of operator can be found.

   Parameters 
   ---------

   (input)
   n      dimension of the eigenproblem matrix B
   wptr   array of pointers that point to work space
   j      starting index for a Lanczos run
   eps    machine epsilon (relative precision)

   (output)
   wptr   array of pointers that point to work space that contains
	  r[j], q[j], q[j-1], p[j], p[j-1]
   ierr   error flag (nonzero if no starting vector can be found)

   Functions used
   --------------

   BLAS		svd_ddot, svd_dcopy, svd_daxpy
   USER		svd_opb, store
   MISC		random

 ***********************************************************************/

double startv(SMat A, double *wptr[], long step, long n) {
	double rnm2, *r, t;
	long irand;
	long id, i;

	/* get initial vector; default is random */
	rnm2 = svd_ddot(n, wptr[0], 1, wptr[0], 1);
	irand = 918273 + step;
	r = wptr[0];
	for (id = 0; id < 3; id++) {
		if (id > 0 || step > 0 || rnm2 == 0)
			for (i = 0; i < n; i++) r[i] = svd_random2(&irand);
		svd_dcopy(n, wptr[0], 1, wptr[3], 1);

		/* apply operator to put r in range (essential if m singular) */
		svd_opb(A, wptr[3], wptr[0], OPBTemp);
		svd_dcopy(n, wptr[0], 1, wptr[3], 1);
		rnm2 = svd_ddot(n, wptr[0], 1, wptr[3], 1);
		if (rnm2 > 0.0) break;
	}

	/* fatal error */
	if (rnm2 <= 0.0) {
		ierr = 8192;
		return(-1);
	}
	if (step > 0) {
		for (i = 0; i < step; i++) {
			store(n, RETRQ, i, wptr[5]);
			t = -svd_ddot(n, wptr[3], 1, wptr[5], 1);
			svd_daxpy(n, t, wptr[5], 1, wptr[0], 1);
		}

		/* make sure q[step] is orthogonal to q[step-1] */
		t = svd_ddot(n, wptr[4], 1, wptr[0], 1);
		svd_daxpy(n, -t, wptr[2], 1, wptr[0], 1);
		svd_dcopy(n, wptr[0], 1, wptr[3], 1);
		t = svd_ddot(n, wptr[3], 1, wptr[0], 1);
		if (t <= eps * rnm2) t = 0.0;
		rnm2 = t;
	}
	return(sqrt(rnm2));
}

/***********************************************************************
 *                                                                     *
 *			error_bound()                                  *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

   Description
   -----------

   Function massages error bounds for very close ritz values by placing 
   a gap between them.  The error bounds are then refined to reflect 
   this.


   Arguments 
   ---------

   (input)
   endl     left end of interval containing unwanted eigenvalues
   endr     right end of interval containing unwanted eigenvalues
   ritz     array to store the ritz values
   bnd      array to store the error bounds
   enough   stop flag


   Functions used
   --------------

   BLAS		svd_idamax
   UTILITY	svd_dmin

 ***********************************************************************/

long error_bound(long *enough, double endl, double endr,
	double *ritz, double *bnd, long step, double tol) {
	long mid, i, neig;
	double gapl, gap;

	/* massage error bounds for very close ritz values */
	mid = svd_idamax(step + 1, bnd, 1);

	for (i = ((step + 1) + (step - 1)) / 2; i >= mid + 1; i -= 1)
		if (fabs(ritz[i - 1] - ritz[i]) < eps34 * fabs(ritz[i]))
			if (bnd[i] > tol && bnd[i - 1] > tol) {
				bnd[i - 1] = sqrt(bnd[i] * bnd[i] + bnd[i - 1] * bnd[i - 1]);
				bnd[i] = 0.0;
			}


	for (i = ((step + 1) - (step - 1)) / 2; i <= mid - 1; i += 1)
		if (fabs(ritz[i + 1] - ritz[i]) < eps34 * fabs(ritz[i]))
			if (bnd[i] > tol && bnd[i + 1] > tol) {
				bnd[i + 1] = sqrt(bnd[i] * bnd[i] + bnd[i + 1] * bnd[i + 1]);
				bnd[i] = 0.0;
			}

	/* refine the error bounds */
	neig = 0;
	gapl = ritz[step] - ritz[0];
	for (i = 0; i <= step; i++) {
		gap = gapl;
		if (i < step) gapl = ritz[i + 1] - ritz[i];
		gap = svd_dmin(gap, gapl);
		if (gap > bnd[i]) bnd[i] = bnd[i] * (bnd[i] / gap);
		if (bnd[i] <= 16.0 * eps * fabs(ritz[i])) {
			neig++;
			if (!*enough) *enough = endl < ritz[i] && ritz[i] < endr;
		}
	}
	return neig;
}

/***********************************************************************
 *                                                                     *
 *				imtqlb()			       *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

   Description
   -----------

   imtqlb() is a translation of a Fortran version of the Algol
   procedure IMTQL1, Num. Math. 12, 377-383(1968) by Martin and 
   Wilkinson, as modified in Num. Math. 15, 450(1970) by Dubrulle.  
   Handbook for Auto. Comp., vol.II-Linear Algebra, 241-248(1971).  
   See also B. T. Smith et al, Eispack Guide, Lecture Notes in 
   Computer Science, Springer-Verlag, (1976).

   The function finds the eigenvalues of a symmetric tridiagonal
   matrix by the implicit QL method.


   Arguments 
   ---------

   (input)
   n      order of the symmetric tridiagonal matrix                   
   d      contains the diagonal elements of the input matrix           
   e      contains the subdiagonal elements of the input matrix in its
          last n-1 positions.  e[0] is arbitrary	             

   (output)
   d      contains the eigenvalues in ascending order.  if an error
            exit is made, the eigenvalues are correct and ordered for
            indices 0,1,...ierr, but may not be the smallest eigenvalues.
   e      has been destroyed.					    
   ierr   set to zero for normal return, j if the j-th eigenvalue has
            not been determined after 30 iterations.		    

   Functions used
   --------------

   UTILITY	svd_fsign
   MISC		svd_pythag

 ***********************************************************************/

void imtqlb(long n, double d[], double e[], double bnd[])

{
   long last, l, m, i, iteration;

   /* various flags */
   long exchange, convergence, underflow;	

   double b, test, g, r, s, c, p, f;

   if (n == 1) return;
   ierr = 0;
   bnd[0] = 1.0;
   last = n - 1;
   for (i = 1; i < n; i++) {
      bnd[i] = 0.0;
      e[i-1] = e[i];
   }
   e[last] = 0.0;
   for (l = 0; l < n; l++) {
      iteration = 0;
	  while (iteration <= 30) {
		  for (m = l; m < n; m++) {
			  convergence = FALSE;
			  if (m == last) break;
			  else {
				  test = fabs(d[m]) + fabs(d[m + 1]);
				  if (test + fabs(e[m]) == test) convergence = TRUE;
			  }
			  if (convergence) break;
		  }
		  p = d[l];
		  f = bnd[l];
		  if (m != l) {
			  if (iteration == 30) {
				  ierr = l;
				  return;
			  }
			  iteration += 1;
			  /*........ form shift ........*/
			  g = (d[l + 1] - p) / (2.0 * e[l]);
			  r = svd_pythag(g, 1.0);
			  g = d[m] - p + e[l] / (g + svd_fsign(r, g));
			  s = 1.0;
			  c = 1.0;
			  p = 0.0;
			  underflow = FALSE;
			  i = m - 1;
			  while (underflow == FALSE && i >= l) {
				  f = s * e[i];
				  b = c * e[i];
				  r = svd_pythag(f, g);
				  e[i + 1] = r;
				  if (r == 0.0) underflow = TRUE;
				  else {
					  s = f / r;
					  c = g / r;
					  g = d[i + 1] - p;
					  r = (d[i] - g) * s + 2.0 * c * b;
					  p = s * r;
					  d[i + 1] = g + p;
					  g = c * r - b;
					  f = bnd[i + 1];
					  bnd[i + 1] = s * bnd[i] + c * f;
					  bnd[i] = c * bnd[i] - s * f;
					  i--;
				  }
			  }       /* end while (underflow != FALSE && i >= l) */
			  /*........ recover from underflow .........*/
			  if (underflow) {
				  d[i + 1] -= p;
				  e[m] = 0.0;
			  }
			  else {
				  d[l] -= p;
				  e[l] = g;
				  e[m] = 0.0;
			  }
		  } 		       		   /* end if (m != l) */
		  else {

			  /* order the eigenvalues */
			  exchange = TRUE;
			  if (l != 0) {
				  i = l;
				  while (i >= 1 && exchange == TRUE) {
					  if (p < d[i - 1]) {
						  d[i] = d[i - 1];
						  bnd[i] = bnd[i - 1];
						  i--;
					  }
					  else exchange = FALSE;
				  }
			  }
			  if (exchange) i = 0;
			  d[i] = p;
			  bnd[i] = f;
			  iteration = 31;
		  }
	  }			       /* end while (iteration <= 30) */
   }				   /* end for (l=0; l<n; l++) */
   return;
}						  /* end main */

/***********************************************************************
 *                                                                     *
 *				imtql2()			       *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

   Description
   -----------

   imtql2() is a translation of a Fortran version of the Algol
   procedure IMTQL2, Num. Math. 12, 377-383(1968) by Martin and 
   Wilkinson, as modified in Num. Math. 15, 450(1970) by Dubrulle.  
   Handbook for Auto. Comp., vol.II-Linear Algebra, 241-248(1971).  
   See also B. T. Smith et al, Eispack Guide, Lecture Notes in 
   Computer Science, Springer-Verlag, (1976).

   This function finds the eigenvalues and eigenvectors of a symmetric
   tridiagonal matrix by the implicit QL method.


   Arguments
   ---------

   (input)                                                             
   nm     row dimension of the symmetric tridiagonal matrix           
   n      order of the matrix                                        
   d      contains the diagonal elements of the input matrix        
   e      contains the subdiagonal elements of the input matrix in its
            last n-1 positions.  e[0] is arbitrary	             
   z      contains the identity matrix				    
                                                                   
   (output)                                                       
   d      contains the eigenvalues in ascending order.  if an error
            exit is made, the eigenvalues are correct but unordered for
            for indices 0,1,...,ierr.				   
   e      has been destroyed.					  
   z      contains orthonormal eigenvectors of the symmetric   
            tridiagonal (or full) matrix.  if an error exit is made,
            z contains the eigenvectors associated with the stored 
          eigenvalues.					
   ierr   set to zero for normal return, j if the j-th eigenvalue has
            not been determined after 30 iterations.		    


   Functions used
   --------------
   UTILITY	svd_fsign
   MISC		svd_pythag

 ***********************************************************************/

void imtql2(long nm, long n, double d[], double e[], double z[])
{
	long index, nnm, j, last, l, m, i, k, iteration, convergence, underflow;
	double b, test, g, r, s, c, p, f;
	if (n == 1) return;
	ierr = 0;
	last = n - 1;
	for (i = 1; i < n; i++) e[i - 1] = e[i];
	e[last] = 0.0;
	nnm = n * nm;
	for (l = 0; l < n; l++) {
		iteration = 0;

		/* look for small sub-diagonal element */
		while (iteration <= 30) {
			for (m = l; m < n; m++) {
				convergence = FALSE;
				if (m == last) break;
				else {
					test = fabs(d[m]) + fabs(d[m + 1]);
					if (test + fabs(e[m]) == test) convergence = TRUE;
				}
				if (convergence) break;
			}
			if (m != l) {

				/* set error -- no convergence to an eigenvalue after
				 * 30 iterations. */
				if (iteration == 30) {
					ierr = l;
					return;
				}
				p = d[l];
				iteration += 1;

				/* form shift */
				g = (d[l + 1] - p) / (2.0 * e[l]);
				r = svd_pythag(g, 1.0);
				g = d[m] - p + e[l] / (g + svd_fsign(r, g));
				s = 1.0;
				c = 1.0;
				p = 0.0;
				underflow = FALSE;
				i = m - 1;
				while (underflow == FALSE && i >= l) {
					f = s * e[i];
					b = c * e[i];
					r = svd_pythag(f, g);
					e[i + 1] = r;
					if (r == 0.0) underflow = TRUE;
					else {
						s = f / r;
						c = g / r;
						g = d[i + 1] - p;
						r = (d[i] - g) * s + 2.0 * c * b;
						p = s * r;
						d[i + 1] = g + p;
						g = c * r - b;

						/* form vector */
						for (k = 0; k < nnm; k += n) {
							index = k + i;
							f = z[index + 1];
							z[index + 1] = s * z[index] + c * f;
							z[index] = c * z[index] - s * f;
						}
						i--;
					}
				}   /* end while (underflow != FALSE && i >= l) */
				/*........ recover from underflow .........*/
				if (underflow) {
					d[i + 1] -= p;
					e[m] = 0.0;
				}
				else {
					d[l] -= p;
					e[l] = g;
					e[m] = 0.0;
				}
			}
			else break;
		}		/*...... end while (iteration <= 30) .........*/
	}		/*...... end for (l=0; l<n; l++) .............*/

	/* order the eigenvalues */
	for (l = 1; l < n; l++) {
		i = l - 1;
		k = i;
		p = d[i];
		for (j = l; j < n; j++) {
			if (d[j] < p) {
				k = j;
				p = d[j];
			}
		}
		/* ...and corresponding eigenvectors */
		if (k != i) {
			d[k] = d[i];
			d[i] = p;
			for (j = 0; j < nnm; j += n) {
				p = z[j + i];
				z[j + i] = z[j + k];
				z[j + k] = p;
			}
		}
	}
	return;
}		/*...... end main ............................*/

/***********************************************************************
 *                                                                     *
 *				machar()			       *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

   Description
   -----------

   This function is a partial translation of a Fortran-77 subroutine 
   written by W. J. Cody of Argonne National Laboratory.
   It dynamically determines the listed machine parameters of the
   floating-point arithmetic.  According to the documentation of
   the Fortran code, "the determination of the first three uses an
   extension of an algorithm due to M. Malcolm, ACM 15 (1972), 
   pp. 949-951, incorporating some, but not all, of the improvements
   suggested by M. Gentleman and S. Marovich, CACM 17 (1974), 
   pp. 276-277."  The complete Fortran version of this translation is
   documented in W. J. Cody, "Machar: a Subroutine to Dynamically 
   Determine Determine Machine Parameters," TOMS 14, December, 1988.


   Parameters reported 
   -------------------

   ibeta     the radix for the floating-point representation       
   it        the number of base ibeta digits in the floating-point
               significand					 
   irnd      0 if floating-point addition chops		      
             1 if floating-point addition rounds, but not in the 
                 ieee style					
             2 if floating-point addition rounds in the ieee style
             3 if floating-point addition chops, and there is    
                 partial underflow				
             4 if floating-point addition rounds, but not in the
                 ieee style, and there is partial underflow    
             5 if floating-point addition rounds in the ieee style,
                 and there is partial underflow                   
   machep    the largest negative integer such that              
                 1.0+float(ibeta)**machep .ne. 1.0, except that 
                 machep is bounded below by  -(it+3)          
   negeps    the largest negative integer such that          
                 1.0-float(ibeta)**negeps .ne. 1.0, except that 
                 negeps is bounded below by  -(it+3)	       

 ***********************************************************************/

void machar(long *ibeta, long *it, long *irnd, long *machep, long *negep) {

	volatile double beta, betain, betah, a, b, ZERO, ONE, TWO, temp, tempa,
		temp1;
	long i, itemp;

	ONE = (double)1;
	TWO = ONE + ONE;
	ZERO = ONE - ONE;

	a = ONE;
	temp1 = ONE;
	while (temp1 - ONE == ZERO) {
		a = a + a;
		temp = a + ONE;
		temp1 = temp - a;
		//b += a; /* to prevent icc compiler error */
	}
	b = ONE;
	itemp = 0;
	while (itemp == 0) {
		b = b + b;
		temp = a + b;
		itemp = (long)(temp - a);
	}
	*ibeta = itemp;
	beta = (double)*ibeta;

	*it = 0;
	b = ONE;
	temp1 = ONE;
	while (temp1 - ONE == ZERO) {
		*it = *it + 1;
		b = b * beta;
		temp = b + ONE;
		temp1 = temp - b;
	}
	*irnd = 0;
	betah = beta / TWO;
	temp = a + betah;
	if (temp - a != ZERO) *irnd = 1;
	tempa = a + beta;
	temp = tempa + betah;
	if ((*irnd == 0) && (temp - tempa != ZERO)) *irnd = 2;

	*negep = *it + 3;
	betain = ONE / beta;
	a = ONE;
	for (i = 0; i < *negep; i++) a = a * betain;
	b = a;
	temp = ONE - a;
	while (temp - ONE == ZERO) {
		a = a * beta;
		*negep = *negep - 1;
		temp = ONE - a;
	}
	*negep = -(*negep);

	*machep = -(*it) - 3;
	a = b;
	temp = ONE + a;
	while (temp - ONE == ZERO) {
		a = a * beta;
		*machep = *machep + 1;
		temp = ONE + a;
	}
	eps = a;
	return;
}

/***********************************************************************
 *                                                                     *
 *                     store()                                         *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

   Description
   -----------

   store() is a user-supplied function which, based on the input
   operation flag, stores to or retrieves from memory a vector.


   Arguments 
   ---------

   (input)
   n       length of vector to be stored or retrieved
   isw     operation flag:
	     isw = 1 request to store j-th Lanczos vector q(j)
	     isw = 2 request to retrieve j-th Lanczos vector q(j)
	     isw = 3 request to store q(j) for j = 0 or 1
	     isw = 4 request to retrieve q(j) for j = 0 or 1
   s	   contains the vector to be stored for a "store" request 

   (output)
   s	   contains the vector retrieved for a "retrieve" request 

   Functions used
   --------------

   BLAS		svd_dcopy

 ***********************************************************************/

void store(long n, long isw, long j, double *s) {
	/* printf("called store %ld %ld\n", isw, j); */
	switch (isw) {
	case STORQ:
		if (!LanStore[j + MAXLL]) {
			if (!(LanStore[j + MAXLL] = svd_doubleArray(n, FALSE, "LanStore[j]")))
				svd_fatalError("svdLAS2: failed to allocate LanStore[%d]", j + MAXLL);
		}
		svd_dcopy(n, s, 1, LanStore[j + MAXLL], 1);
		break;
	case RETRQ:
		if (!LanStore[j + MAXLL])
			svd_fatalError("svdLAS2: store (RETRQ) called on index %d (not allocated)",
				j + MAXLL);
		svd_dcopy(n, LanStore[j + MAXLL], 1, s, 1);
		break;
	case STORP:
		if (j >= MAXLL) {
			svd_error("svdLAS2: store (STORP) called with j >= MAXLL");
			break;
		}
		if (!LanStore[j]) {
			if (!(LanStore[j] = svd_doubleArray(n, FALSE, "LanStore[j]")))
				svd_fatalError("svdLAS2: failed to allocate LanStore[%d]", j);
		}
		svd_dcopy(n, s, 1, LanStore[j], 1);
		break;
	case RETRP:
		if (j >= MAXLL) {
			svd_error("svdLAS2: store (RETRP) called with j >= MAXLL");
			break;
		}
		if (!LanStore[j])
			svd_fatalError("svdLAS2: store (RETRP) called on index %d (not allocated)",
				j);
		svd_dcopy(n, LanStore[j], 1, s, 1);
		break;
	}
	return;
}

static long imin(long a, long b) {return (a < b) ? a : b;}

static void debug(char *fmt, ...) {
	va_list args;
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	va_end(args);
}

static void fatalError(char *fmt, ...) {
	va_list args;
	va_start(args, fmt);
	fprintf(stderr, "ERROR: ");
	vfprintf(stderr, fmt, args);
	fprintf(stderr, "\a\n");
	va_end(args);
	exit(1);
}

COLLOC* NS_read_input_file(FILE *fid, int *num_rec)
{
	fseek(fid, 0, SEEK_END);
	long file_size = ftell(fid);
	(*num_rec) = file_size / sizeof(COLLOC);
	COLLOC *cooccur = (COLLOC*)malloc(sizeof(COLLOC) * (*num_rec));
	if (cooccur == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	rewind(fid);
	if (fread(cooccur, sizeof(COLLOC), *num_rec, fid) == (*num_rec))
	{
		rewind(fid);
		return cooccur;
	}
	else
	{
		fclose(fid);
		exit(EXIT_FAILURE);
	}
}

void NS_get_rowcol_make0ind(COLLOC *cooccur, int num_rec, int *num_row, int *num_col, double *max_val, double *average)
{
	int i;
	*num_row = -1;
	*num_col = -1;
	*max_val = -DBL_MAX;
	*average = 0;
	for (i = 0; i < num_rec; i++)
	{
		if (cooccur[i].row > *num_row)
			*num_row = cooccur[i].row;
		if (cooccur[i].col > *num_col)
			*num_col = cooccur[i].col;
		if (cooccur[i].val > *max_val)
			*max_val = cooccur[i].val;

		*average += cooccur[i].val;
		cooccur[i].row--;
		cooccur[i].col--;
	}
	*average /= num_rec;
}

void NS_calc_indices_rowmajor(COLLOC *cooccur, int num_rec, int num_row, int *start_index)
{
	int i, last_row = -1;
	for (i = 0; i < num_rec; i++)
	{
		if (cooccur[i].row != last_row)
		{
			last_row++;
			while (cooccur[i].row != last_row)
			{
				start_index[last_row] = i;
				last_row++;
			}
			start_index[cooccur[i].row] = i;
		}
	}
	start_index[num_row] = num_rec;
}

void NS_calc_indices_colmajor(COLLOC *cooccur, int num_rec, int num_col, int *start_index)
{
	int i, last_col = -1;
	for (i = 0; i < num_rec; i++)
	{
		if (cooccur[i].col != last_col)
		{
			last_col++;
			while (cooccur[i].col != last_col)
			{
				start_index[last_col] = i;
				last_col++;
			}
			start_index[cooccur[i].col] = i;
		}
	}
	start_index[num_col] = num_rec;
}

int NS_compare_colloc_rowmajor(const void *a, const void *b) {
	int diff;
	if ((diff = ((COLLOC*)a)->row - ((COLLOC*)b)->row) != 0)
		return diff;
	else
		return (((COLLOC*)a)->col - ((COLLOC*)b)->col);
}

int NS_compare_colloc_colmajor(const void *a, const void *b) {
	int diff;
	if ((diff = ((COLLOC*)a)->col - ((COLLOC*)b)->col) != 0)
		return diff;
	else
		return (((COLLOC*)a)->row - ((COLLOC*)b)->row);
}

SMat NS_convertCOLLOC2SMat_rowmajor(COLLOC *cooccur, int rows, int cols, int vals, int *start_index)
{
	int c, i, v;
	SMat S = svdNewSMat(cols, rows, vals);
	if (!S) return NULL;

	for (c = 0, v = 0; c < rows; c++) {
		S->pointr[c] = v;
		for (i = start_index[c]; i < start_index[c + 1]; i++, v++) {
			S->rowind[v] = cooccur[i].col;
			S->value[v] = cooccur[i].val;
		}
	}
	S->pointr[rows] = vals;
	return S;
}

SMat NS_convertCOLLOC2SMat_colmajor(COLLOC *cooccur, int rows, int cols, int vals, int *start_index)
{
	int c, i, v;
	SMat S = svdNewSMat(rows, cols, vals);
	if (!S) return NULL;

	for (c = 0, v = 0; c < cols; c++) {
		S->pointr[c] = v;
		for (i = start_index[c]; i < start_index[c + 1]; i++, v++) {
			S->rowind[v] = cooccur[i].row;
			S->value[v] = cooccur[i].val;
		}
	}
	S->pointr[cols] = vals;
	return S;
}

// on 32-bit system, maximum possible number to get is: 1,073,741,824 
int NS_randi_uniform()
{
	if (RAND_MAX >= 32768)
		return rand();
	else
		return ((rand() << 15) | rand());
}

int NS_randi_uniform_limit(int max)
{
	return (NS_randi_uniform() % max);
}

double NS_randf_uniform()
{
	if (RAND_MAX >= 32768)
		return (double)((double)rand() / (double)RAND_MAX);
	else
		return (double)((double)((rand() << 15) | rand()) / (double)((RAND_MAX << 15) | RAND_MAX));
}

// Box-Muller transform
double NS_randf_normal(double mu, double sigma)
{
	double two_pi = 2.0 * 3.14159265358979323846;
	double u1, u2, z0;
	u2 = NS_randf_uniform();
	do
	{
		u1 = NS_randf_uniform();
	} while (u1 <= DBL_MIN);

	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	return z0 * sigma + mu;
}

int NS_randi_unigram(double *cumprob, int cnt)
{
	int i;
	double rnd = NS_randf_uniform();
	for (i = 0; i < cnt; i++)
		if (rnd <= cumprob[i])
			return i;
	return (cnt - 1);
}


double NS_dot_product(COLLOC *cooccur, int is, int ie, int js, int je)
{
	double dot = 0.0;
	while (is < ie && js < je)
	{
		if (cooccur[is].row < cooccur[js].row)
			is++;
		else if (cooccur[is].row > cooccur[js].row)
			js++;
		else
		{
			dot += cooccur[is].val * cooccur[js].val;
			is++;
			js++;
		}
	}
	return dot;
}

double NS_cos_theta_rowmajor(COLLOC *cooccur, int is, int ie, int js, int je)
{
	double res = 0.0, len1 = 0.0, len2 = 0.0;
	while (is < ie && js < je)
	{
		if (cooccur[is].col < cooccur[js].col)
		{
			len1 += cooccur[is].val * cooccur[is].val;
			is++;
		}
		else if (cooccur[is].col > cooccur[js].col)
		{
			len2 += cooccur[js].val * cooccur[js].val;
			js++;
		}
		else
		{
			res += cooccur[is].val * cooccur[js].val;
			len1 += cooccur[is].val * cooccur[is].val;
			len2 += cooccur[js].val * cooccur[js].val;
			is++;
			js++;
		}
	}
	while (is < ie)
	{
		len1 += cooccur[is].val * cooccur[is].val;
		is++;
	}
	while (js < je)
	{
		len2 += cooccur[js].val * cooccur[js].val;
		js++;
	}
	if (len1 > 0 && len2 > 0)
		res /= sqrt(len1) * sqrt(len2);
	return res;
}

double NS_cos_theta_colmajor(COLLOC *cooccur, int is, int ie, int js, int je)
{
	double res = 0.0, len1 = 0.0, len2 = 0.0;
	while (is < ie && js < je)
	{
		if (cooccur[is].row < cooccur[js].row)
		{
			len1 += cooccur[is].val * cooccur[is].val;
			is++;
		}
		else if (cooccur[is].row > cooccur[js].row)
		{
			len2 += cooccur[js].val * cooccur[js].val;
			js++;
		}
		else
		{
			res += cooccur[is].val * cooccur[js].val;
			len1 += cooccur[is].val * cooccur[is].val;
			len2 += cooccur[js].val * cooccur[js].val;
			is++;
			js++;
		}
	}
	while (is < ie)
	{
		len1 += cooccur[is].val * cooccur[is].val;
		is++;
	}
	while (js < je)
	{
		len2 += cooccur[js].val * cooccur[js].val;
		js++;
	}
	if (len1 > 0 && len2 > 0)
		res /= sqrt(len1) * sqrt(len2);
	return res;
}

void NS_calc_unigram_cumprob_smooth(double *sum_col, int cnt, double alpha)
{
	int i;
	double sum = 0.0;
	for (i = 0; i < cnt; i++)
	{
		sum_col[i] = pow(sum_col[i], alpha);
		sum += sum_col[i];
	}

	sum_col[0] /= sum;
	for (i = 1; i < cnt; i++)
		sum_col[i] = (sum_col[i] / sum) + sum_col[i - 1];
}

// assumes neg_vals is even.
void NS_generate_unigram_negative_samples(COLLOC * cooccur, int *start_index, COLLOC *neg_samples, int rows, int cols, int *neg_vals, double *cumprob)
{
	int i, n_attempt, n_half, n_neg_row;
	n_half = (*neg_vals) / 2;
	n_neg_row = n_half / rows;
	n_half = rows * n_neg_row;
	(*neg_vals) = 2 * n_half;

#pragma omp parallel default(none) private(i, n_attempt) shared(n_half, n_neg_row, cooccur, start_index, neg_samples, cumprob, cols)
	{
#pragma omp for
		for (i = 0; i < n_half; i++)
		{
			neg_samples[i].row = i / n_neg_row;
			neg_samples[i].col = NS_randi_unigram(cumprob, cols);
			neg_samples[i].val = -3.0;// +NS_cos_theta_rowmajor(cooccur, start_index[neg_samples[i].row], start_index[neg_samples[i].row + 1], start_index[neg_samples[i].col], start_index[neg_samples[i].col + 1]);

			n_attempt = 0;
			while (n_attempt < 5 && NS_cos_theta_rowmajor(cooccur, start_index[neg_samples[i].row], start_index[neg_samples[i].row + 1], start_index[neg_samples[i].col], start_index[neg_samples[i].col + 1]) > 0.001)
			{
				n_attempt++;
				neg_samples[i].col = NS_randi_unigram(cumprob, cols);
			}

			neg_samples[i + n_half].row = neg_samples[i].col;
			neg_samples[i + n_half].col = neg_samples[i].row;
			neg_samples[i + n_half].val = neg_samples[i].val;
		}
	}
}

// assumes both cooccurrences and negative samples are sorted (rowmajor)
int NS_validate_negative_samples_rowmajor(COLLOC *cooccur, COLLOC *neg_samples, int rows, int cols, int vals, int neg_vals, char *neg_valid)
{
	int n_valid = 0, i1 = 0, i2;
	for (i2 = 0; i2 < neg_vals; i2++)
	{
		if (neg_samples[i2].row == neg_samples[i2].col)
			neg_valid[i2] = 0;
		else if (i2 > 0 && neg_samples[i2].row == neg_samples[i2 - 1].row && neg_samples[i2].col == neg_samples[i2 - 1].col)
			neg_valid[i2] = 0;
		else
		{
			while (i1 < vals && cooccur[i1].row < neg_samples[i2].row)
				i1++;
			while (i1 < vals && cooccur[i1].row == neg_samples[i2].row && cooccur[i1].col < neg_samples[i2].col)
				i1++;

			if (i1 < vals && cooccur[i1].row == neg_samples[i2].row && cooccur[i1].col == neg_samples[i2].col)
				neg_valid[i2] = 0;
			else
			{
				neg_valid[i2] = 1;
				n_valid++;
			}
		}
	}
	return n_valid;
}

// assumes both cooccurrences and negative samples are sorted (colmajor)
int NS_validate_negative_samples_colmajor(COLLOC *cooccur, COLLOC *neg_samples, int rows, int cols, int vals, int neg_vals, char *neg_valid)
{
	int n_valid = 0, i1 = 0, i2;
	for (i2 = 0; i2 < neg_vals; i2++)
	{
		if (neg_samples[i2].row == neg_samples[i2].col)
			neg_valid[i2] = 0;
		else if (i2 > 0 && neg_samples[i2].col == neg_samples[i2 - 1].col && neg_samples[i2].row == neg_samples[i2 - 1].row)
			neg_valid[i2] = 0;
		else
		{
			while (i1 < vals && cooccur[i1].col < neg_samples[i2].col)
				i1++;
			while (i1 < vals && cooccur[i1].col == neg_samples[i2].col && cooccur[i1].row < neg_samples[i2].row)
				i1++;

			if (i1 < vals && cooccur[i1].col == neg_samples[i2].col && cooccur[i1].row == neg_samples[i2].row)
				neg_valid[i2] = 0;
			else
			{
				neg_valid[i2] = 1;
				n_valid++;
			}
		}
	}
	return n_valid;
}

// in rowmajor implementation of this function, we actually write transpose of cooccur into S but since it is symmetric no problem!
SMat NS_mergePosNeg_into_SMat_rowmajor(COLLOC *cooccur, COLLOC *neg_samples, int rows, int cols, int vals, int neg_vals, int n_valid, char *neg_valid, double *avg_pos, double *avg_neg)
{
	(*avg_pos) = 0.0;
	(*avg_neg) = 0.0;

	int c, v, i1 = 0, i2 = 0;
	SMat S = svdNewSMat(cols, rows, vals + n_valid);
	if (!S) return NULL;

	for (c = 0, v = 0; c < rows; c++)
	{
		S->pointr[c] = v;
		while ((i1 < vals && cooccur[i1].row <= c) || (i2 < neg_vals && neg_samples[i2].row <= c))
		{
			if (i2 < neg_vals && neg_valid[i2] == 0)
				i2++;
			else if (i1 >= vals || cooccur[i1].row > c)
			{
				S->rowind[v] = neg_samples[i2].col;
				S->value[v] = neg_samples[i2].val;
				(*avg_neg) += neg_samples[i2].val;
				v++;
				i2++;
			}
			else if (i2 >= neg_vals || neg_samples[i2].row > c)
			{
				S->rowind[v] = cooccur[i1].col;
				S->value[v] = cooccur[i1].val;
				(*avg_pos) += cooccur[i1].val;
				v++;
				i1++;
			}
			else if (cooccur[i1].col < neg_samples[i2].col)
			{
				S->rowind[v] = cooccur[i1].col;
				S->value[v] = cooccur[i1].val;
				(*avg_pos) += cooccur[i1].val;
				v++;
				i1++;
			}
			else
			{
				S->rowind[v] = neg_samples[i2].col;
				S->value[v] = neg_samples[i2].val;
				(*avg_neg) += neg_samples[i2].val;
				v++;
				i2++;
			}
		}
	}
	S->pointr[rows] = vals + n_valid;

	(*avg_pos) /= vals;
	(*avg_neg) /= n_valid;
	return S;
}

SMat NS_mergePosNeg_into_SMat_colmajor(COLLOC *cooccur, COLLOC *neg_samples, int rows, int cols, int vals, int neg_vals, int n_valid, char *neg_valid, double *avg_pos, double *avg_neg)
{
	(*avg_pos) = 0.0;
	(*avg_neg) = 0.0;

	int c, v, i1 = 0, i2 = 0;
	SMat S = svdNewSMat(rows, cols, vals + n_valid);
	if (!S) return NULL;

	for (c = 0, v = 0; c < cols; c++)
	{
		S->pointr[c] = v;
		while ((i1 < vals && cooccur[i1].col <= c) || (i2 < neg_vals && neg_samples[i2].col <= c))
		{
			if (i2 < neg_vals && neg_valid[i2] == 0)
				i2++;
			else if (i1 >= vals || cooccur[i1].col > c)
			{
				S->rowind[v] = neg_samples[i2].row;
				S->value[v] = neg_samples[i2].val;
				(*avg_neg) += neg_samples[i2].val;
				v++;
				i2++;
			}
			else if (i2 >= neg_vals || neg_samples[i2].col > c)
			{
				S->rowind[v] = cooccur[i1].row;
				S->value[v] = cooccur[i1].val;
				(*avg_pos) += cooccur[i1].val;
				v++;
				i1++;
			}
			else if (cooccur[i1].row < neg_samples[i2].row)
			{
				S->rowind[v] = cooccur[i1].row;
				S->value[v] = cooccur[i1].val;
				(*avg_pos) += cooccur[i1].val;
				v++;
				i1++;
			}
			else
			{
				S->rowind[v] = neg_samples[i2].row;
				S->value[v] = neg_samples[i2].val;
				(*avg_neg) += neg_samples[i2].val;
				v++;
				i2++;
			}
		}
	}
	S->pointr[cols] = vals + n_valid;

	(*avg_pos) /= vals;
	(*avg_neg) /= n_valid;
	return S;
}

void NS_calc_stats(COLLOC *cooccur, int nrow, int ncol, int nval, double *sum_row, double *sum_col, double *sumsum)
{
	int i;
	*sumsum = 0;
	for (i = 0; i < nrow; i++)
		sum_row[i] = 0.0;
	for (i = 0; i < ncol; i++)
		sum_col[i] = 0.0;

	for (i = 0; i < nval; i++)
	{
		sum_row[cooccur[i].row] += cooccur[i].val;
		sum_col[cooccur[i].col] += cooccur[i].val;
		*sumsum += cooccur[i].val;
	}
}

COLLOC *NS_pmi(COLLOC *cooccur, int nval, int takePMI, double pmiThreshold, double pmiShift, double *sum_row, double *sum_col, double sumsum, int rows, int cols, int contextDistSmooth, double alpha, double *minpmi, double *maxpmi, int *n_valid)
{
	int i, j;
	double d_temp;

	double *smooth1 = (double*)malloc(sizeof(double) * cols);
	double *smooth2 = (double*)malloc(sizeof(double) * rows);
	if (smooth1 == NULL || smooth2 == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	for (i = 0, d_temp = 0.0; i < cols; i++)
	{
		if (contextDistSmooth > 0)
			smooth1[i] = pow(sum_col[i], alpha);
		else
			smooth1[i] = sum_col[i];
		d_temp += smooth1[i];
	}
	for (i = 0; i < cols; i++)
		smooth1[i] /= d_temp;

	for (i = 0, d_temp = 0.0; i < rows; i++)
	{
		if (contextDistSmooth > 1)
			smooth2[i] = pow(sum_row[i], alpha);
		else
			smooth2[i] = sum_row[i];
		d_temp += smooth2[i];
	}
	for (i = 0; i < rows; i++)
		smooth2[i] /= d_temp;

	*n_valid = 0;
	*minpmi = DBL_MAX;
	*maxpmi = -DBL_MAX;
	for (i = 0; i < nval; i++)
	{
		if (takePMI == 2)
			cooccur[i].val = log2((cooccur[i].val) / (sumsum * smooth2[cooccur[i].row] * smooth1[cooccur[i].col]));
		else
			cooccur[i].val = log10((cooccur[i].val) / (sumsum * smooth2[cooccur[i].row] * smooth1[cooccur[i].col]));

		if (cooccur[i].val > pmiThreshold)
			(*n_valid)++;

		if (cooccur[i].val < (*minpmi))
			*minpmi = cooccur[i].val;
		if (cooccur[i].val >(*maxpmi))
			*maxpmi = cooccur[i].val;
	}
	free(smooth1); smooth1 = NULL;
	free(smooth2); smooth2 = NULL;

	if ((*n_valid) == nval)
	{
		for (i = 0; i < nval; i++)
			cooccur[i].val += pmiShift;

		return cooccur;
	}
	else
	{
		COLLOC *pmi_table = (COLLOC*)malloc(sizeof(COLLOC) * (*n_valid));
		if (pmi_table == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

		for (i = 0, j = 0; i < nval; i++)
		{
			if (cooccur[i].val > pmiThreshold)
			{
				pmi_table[j].row = cooccur[i].row;
				pmi_table[j].col = cooccur[i].col;
				pmi_table[j].val = cooccur[i].val + pmiShift;
				j++;
			}
		}

		free(cooccur); cooccur = NULL;
		return pmi_table;
	}
}

int NS_is_already_sorted(COLLOC *cooccur, int vals)
{
	int i, dir, prev_dir = 0;
	for (i = 1; i < vals; i++)
	{
		dir = SORT_FUNC(&cooccur[i], &cooccur[i - 1]);
		if (dir != 0)
		{
			if ((prev_dir > 0 && dir < 0) || (prev_dir < 0 && dir > 0))
				return FALSE;
			prev_dir = dir;
		}
	}
	return TRUE;
}

SMat NS_loadPrepareMatrix(char *filename, int takeLogBefore, int takeSqrtBefore, int takePMI, double pmiThreshold, double pmiShift, int contextDistSmooth, int takeLog, int takeSqrt, double neg_ratio, double alpha, double truncate, int verbose) {
	int i, rows = 0, cols = 0, vals = 0, neg_vals = 0, n_valid_neg = 0, n_valid_pmi = 0;
	double max_val = 0.0, average = 0.0, average_neg = 0.0, sumsum = 0.0, min_pmi = 0.0, max_pmi = 0.0;

	if (verbose > 0) printf("Reading input file...\n");
	FILE *file = svd_fatalReadFile(filename);
	COLLOC *cooccur = NS_read_input_file(file, &vals);
	fclose(file);

	NS_get_rowcol_make0ind(cooccur, vals, &rows, &cols, &max_val, &average);
	if (NS_is_already_sorted(cooccur, vals) == 0)
	{
		if (verbose > 0) printf("Sorting the entries...\n");
		qsort(cooccur, vals, sizeof(COLLOC), SORT_FUNC);
	}
	else if (verbose > 0) printf("Entries were already sorted...\n");

	double *sum_row = (double*)malloc(sizeof(double) * rows);
	double *sum_col = (double*)malloc(sizeof(double) * cols);
	if (sum_row == NULL || sum_col == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	if (neg_ratio > 0.0 || (takePMI > 0 && takeLogBefore == 0 && takeSqrtBefore == 0))
	{
		if (verbose > 0) printf("Calculating some stats...\n");
		NS_calc_stats(cooccur, rows, cols, vals, sum_row, sum_col, &sumsum);
	}

	if (takePMI)
	{
		if (takeLogBefore || takeSqrtBefore)
		{
			double sumsum_trans = 0.0;
			double *sum_row_trans = (double*)malloc(sizeof(double) * rows);
			double *sum_col_trans = (double*)malloc(sizeof(double) * cols);
			if (sum_row_trans == NULL || sum_col_trans == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

			if (takeLogBefore)
			{
				if (verbose > 0) printf("Taking log of the entries before applying PMI...\n");
				for (i = 0; i < vals; i++)
					cooccur[i].val = log2(cooccur[i].val);
			}
			else if (takeSqrtBefore)
			{
				if (verbose > 0) printf("Taking sqrt of the entries before applying PMI...\n");
				for (i = 0; i < vals; i++)
					cooccur[i].val = sqrt(cooccur[i].val);
			}

			if (verbose > 0) printf("Calculating some stats for PMI...\n");
			NS_calc_stats(cooccur, rows, cols, vals, sum_row_trans, sum_col_trans, &sumsum_trans);
			if (verbose > 0) printf("Applying PMI...\n");
			cooccur = NS_pmi(cooccur, vals, takePMI, pmiThreshold, pmiShift, sum_row_trans, sum_col_trans, sumsum_trans, rows, cols, contextDistSmooth, alpha, &min_pmi, &max_pmi, &n_valid_pmi);

			free(sum_row_trans); sum_row_trans = NULL;
			free(sum_col_trans); sum_col_trans = NULL;
		}
		else
		{
			if (verbose > 0) printf("Applying PMI...\n");
			cooccur = NS_pmi(cooccur, vals, takePMI, pmiThreshold, pmiShift, sum_row, sum_col, sumsum, rows, cols, contextDistSmooth, alpha, &min_pmi, &max_pmi, &n_valid_pmi);
		}

		if (verbose > 0)
		{
			printf("\n---------- PMI (threshold = %.2f, shift = %.2f) ----------\n", pmiThreshold, pmiShift);
			printf("# of non-zero entries before PMI: %d (%.4f%% density)\n", vals, 100.0 * (double)((double)vals / (double)((double)rows * (double)cols)));
			printf("# of non-zero entries after PMI:  %d (%.4f%% density)\n", n_valid_pmi, 100.0 * (double)((double)n_valid_pmi / (double)((double)rows * (double)cols)));
			printf("Max PMI value: %f\n", max_pmi);
			printf("Min PMI value: %f\n\n", min_pmi);
		}
		vals = n_valid_pmi;
	}

	//FILE *fhigh = fopen("pmi_high.txt", "wt");
	//FILE *flow = fopen("pmi_low.txt", "wt");
	//for (i = 0; i < vals; i++)
	//{
	//	if (cooccur[i].row > 30 && cooccur[i].col > 30)
	//	{
	//		if (cooccur[i].val > 10)
	//			fprintf(fhigh, "%d %d %f\n", (cooccur[i].row + 1), (cooccur[i].col + 1), cooccur[i].val);
	//		if (cooccur[i].val < -6)
	//			fprintf(flow, "%d %d %f\n", (cooccur[i].row + 1), (cooccur[i].col + 1), cooccur[i].val);
	//	}
	//}
	//fclose(fhigh);
	//fclose(flow);

	if (truncate < (DBL_MAX - 1))
	{
		if (verbose > 0) printf("Truncating values to %f before applying SVD...\n", truncate);
		for (i = 0; i < vals; i++)
			if (cooccur[i].val > truncate)
				cooccur[i].val = truncate;
	}

	if (takeLog || (takePMI <= 0 && takeLogBefore))
	{
		if (verbose > 0) printf("Taking log of the entries before applying SVD...\n");
		for (i = 0; i < vals; i++)
		{
			if (cooccur[i].val > DBL_MIN)
				cooccur[i].val = log2(cooccur[i].val);
			else
				cooccur[i].val = -DBL_MAX;  // This situation MUST be avoided by the user by shifting PMI and/or using proper cutoff
		}
	}
	else if (takeSqrt || (takePMI <= 0 && takeSqrtBefore))
	{
		if (verbose > 0) printf("Taking sqrt of the entries before applying SVD...\n");
		for (i = 0; i < vals; i++)
		{
			if (cooccur[i].val >= 0.0)
				cooccur[i].val = sqrt(cooccur[i].val);
		}
	}

	int sz = rows + 1; // cols + 1
	int *start_index = (int*)malloc(sizeof(int) * sz);
	if (start_index == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	NS_calc_indices_rowmajor(cooccur, vals, rows, start_index);

	SMat S;
	if (neg_ratio > 0.0)
	{
		neg_vals = round(neg_ratio * vals * (1.0 + (double)((double)vals / (double)((double)rows * (double)cols))));
		if (neg_vals % 2 == 1) // even number of negative samples (symmetric form)
			neg_vals++;

		COLLOC *neg_samples = (COLLOC*)malloc(sizeof(COLLOC) * neg_vals);
		char *neg_valid = (char*)malloc(sizeof(char) * neg_vals);
		if (neg_samples == NULL || neg_valid == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

		if (verbose > 0) printf("Calculating smoothed cumulative probabilities of unigrams...\n");
		NS_calc_unigram_cumprob_smooth(sum_col, cols, alpha);

		if (verbose > 0) printf("Generating negative samples...\n");
		NS_generate_unigram_negative_samples(cooccur, start_index, neg_samples, rows, cols, &neg_vals, sum_col);
		if (verbose > 0) printf("Sorting negative samples...\n");
		qsort(neg_samples, neg_vals, sizeof(COLLOC), SORT_FUNC);

		if (verbose > 0) printf("Validating negative samples and merging with positives...\n");
		n_valid_neg = NS_validate_negative_samples_rowmajor(cooccur, neg_samples, rows, cols, vals, neg_vals, neg_valid);
		S = NS_mergePosNeg_into_SMat_rowmajor(cooccur, neg_samples, rows, cols, vals, neg_vals, n_valid_neg, neg_valid, &average, &average_neg);

		if (verbose > 0)
		{
			printf("\n---------- Negative Sampling (ratio = %.2f, alpha = %.2f) ----------\n", neg_ratio, alpha);
			printf("Positive count:         %d\n", vals);
			printf("Negative samples count: %d\n", n_valid_neg);
			printf("Total non-zero count:   %d\n", vals + n_valid_neg);
			printf("Average positive value: %f\n", average);
			printf("Average negative value: %f\n", average_neg);
			printf("Density increased from %.4f%% to %.4f%%\n\n", 100.0 * (double)((double)vals / (double)((double)rows * (double)cols)), 100.0 * (double)((double)((double)vals + (double)n_valid_neg) / (double)((double)rows * (double)cols)));
		}

		free(neg_samples); neg_samples = NULL;
		free(neg_valid); neg_valid = NULL;
	}
	else
		S = NS_convertCOLLOC2SMat_rowmajor(cooccur, rows, cols, vals, start_index);

	free(cooccur); cooccur = NULL;
	free(start_index); start_index = NULL;
	free(sum_row); sum_row = NULL;
	free(sum_col); sum_col = NULL;
	return S;
}

void NS_usage(char *progname) {
	debug("SVD Version %s\n"
		"SVD part is written by Douglas Rohde based on code adapted from SVDPACKC\n", SVDVersion);
	debug("Negative sampling and the PMI part is written by Behrouz Haji Soleimani (behrouz.hajisoleimani@dal.ca)\n\n");

	debug("usage: %s [options]\n", progname);
	debug("  -i input_file    path to the input file (it can be either cooccurrence matrix or PMI matrix)\n"
		"  -b vocab_file    path to the vocabulary file\n"
		"  -o output_file   Root of files in which to store resulting U,S,V\n"
		"  -o output_file   Root of files in which to store resulting U,S,V\n"
		"  -d dimensions    Desired SVD triples (default is 100)\n"
		"  -e bound         Minimum magnitude of wanted eigenvalues (1e-30)\n"
		"  -k kappa         Accuracy parameter for las2 (1e-6)\n"
		"  -t no_thread     number of threads to use in parallel processing\n"
		"  -v verbosity     Default 1.  0 for no feedback, 2 for more\n");
	exit(1);
}


void NS_write_output_file(char *vocab_file, char *output_file, DMat matrix)
{
	FILE *fv = NULL;
	if (vocab_file)
	{
		fv = fopen(vocab_file, "rt");
		if (fv == NULL) { printf("Error: could not open vocab file \"%s\".\r\n", vocab_file); exit(EXIT_FAILURE); }
	}

	FILE *fo = fopen(output_file, "wt");
	if (fo == NULL) { printf("Error: could not open output file for writing \"%s\".\r\n", output_file); exit(EXIT_FAILURE); }

	char fmt[32];
	char *word = (char*)malloc(sizeof(char) * 1024);
	if (word == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	sprintf(fmt, "%%%ds", 1024);

	int i, j;
	for (i = 0; i < matrix->rows; i++)
	{
		if (vocab_file)
		{
			if (fscanf(fv, fmt, word) == 0) { fclose(fv); fclose(fo); exit(EXIT_FAILURE); }
			fprintf(fo, "%s", word);
			for (j = 0; j < matrix->cols; j++)
				fprintf(fo, " %.6f", matrix->value[i][j]);
			fprintf(fo, "\n");
			if (fscanf(fv, fmt, word) == 0) { fclose(fv); fclose(fo); exit(EXIT_FAILURE); }
		}
		else
		{
			for (j = 0; j < (matrix->cols - 1); j++)
				fprintf(fo, "%.6f ", matrix->value[i][j]);
			fprintf(fo, "%.6f\n", matrix->value[i][matrix->cols - 1]);
		}
	}

	if (vocab_file)
		fclose(fv);
	fclose(fo);
	free(word); word = NULL;
}


void NS_normalize_rows(DMat A)
{
	int i, j;
	double sum;

	for (i = 0; i < A->rows; i++)
	{
		sum = 0;
		for (j = 0; j < A->cols; j++)
			sum += A->value[i][j] * A->value[i][j];
		sum = sqrt(sum);
		for (j = 0; j < A->cols; j++)
			A->value[i][j] /= sum;
	}
}


void NS_normalize_cols(DMat A)
{
	int i, j;
	double sum;

	for (i = 0; i < A->cols; i++)
	{
		sum = 0;
		for (j = 0; j < A->rows; j++)
			sum += A->value[j][i] * A->value[j][i];
		sum = sqrt(sum);
		for (j = 0; j < A->rows; j++)
			A->value[j][i] /= sum;
	}
}


int NS_strcmp(char *s1, char *s2) {
	while (*s1 != '\0' && *s1 == *s2) { s1++; s2++; }
	return(*s1 - *s2);
}


int NS_parse_arg(char *str, int argc, char **argv, int mandatory) {
	int i;
	for (i = 1; i < argc; i++) {
		if (!NS_strcmp(str, argv[i])) {
			if (mandatory && (i == argc - 1)) {
				printf("No argument given for %s\n", str);
				exit(EXIT_FAILURE);
			}
			return i;
		}
	}
	return -1;
}


int main(int argc, char *argv[]) {
	SVDRec R = NULL;
	SMat A = NULL;

	int i;
	int iterations = 0;
	int dimensions = DEFAULT_DIM;
	int takeLog = 0, takeLogBefore = 0;
	int takeSqrt = 0, takeSqrtBefore = 0;
	int takePMI = 0, contextDistSmooth = 0;
	int writeMode = 3, verbose = 1;
	double pmiThreshold = DEFAULT_CUTP;
	double pmiShift = DEFAULT_SHIFT;
	double las2end[2] = { -1.0e-30, 1.0e-30 };
	double kappa = 1e-6;
	double neg_ratio = -1.0, alpha = 0.75, truncate = DBL_MAX;
	char *vectorFile, *vocabFile, *inputFile;
	vectorFile = (char*)malloc(sizeof(char) * 1024);
	vocabFile = (char*)malloc(sizeof(char) * 1024);
	inputFile = (char*)malloc(sizeof(char) * 1024);
	if (inputFile == NULL || vocabFile == NULL || vectorFile == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	// switches -a -b -cp -d -e -h -i -k -l -l0 -n -o -p -p2 -ps -s -s0 -sh -t -tr -v -w

	if ((i = NS_parse_arg((char *)"-i", argc, argv, 1)) > 0 || (i = NS_parse_arg((char *)"-input", argc, argv, 1)) > 0)
		strcpy(inputFile, argv[i + 1]);
	else
		NS_usage(argv[0]);

	if ((i = NS_parse_arg((char *)"-b", argc, argv, 1)) > 0 || (i = NS_parse_arg((char *)"-vocab", argc, argv, 1)) > 0)
		strcpy(vocabFile, argv[i + 1]);
	else
		NS_usage(argv[0]);

	if ((i = NS_parse_arg((char *)"-o", argc, argv, 1)) > 0 || (i = NS_parse_arg((char *)"-embedding", argc, argv, 1)) > 0)
		strcpy(vectorFile, argv[i + 1]);
	else
		NS_usage(argv[0]);

	if ((i = NS_parse_arg((char *)"-d", argc, argv, 1)) > 0 || (i = NS_parse_arg((char *)"-dimension", argc, argv, 1)) > 0)
	{
		dimensions = atoi(argv[i + 1]);
		if (dimensions < 0)
			dimensions = DEFAULT_DIM;
	}

	if ((i = NS_parse_arg((char *)"-t", argc, argv, 1)) > 0 || (i = NS_parse_arg((char *)"-thread", argc, argv, 1)) > 0)
	{
		no_thread = atoi(argv[i + 1]);
		if (no_thread < 1)
			no_thread = DEFAULT_THREAD;
	}

	if ((i = NS_parse_arg((char *)"-l", argc, argv, 0)) > 0 || (i = NS_parse_arg((char *)"-log", argc, argv, 0)) > 0)
		takeLog = TRUE;

	if ((i = NS_parse_arg((char *)"-s", argc, argv, 0)) > 0 || (i = NS_parse_arg((char *)"-sqrt", argc, argv, 0)) > 0)
		takeSqrt = TRUE;

	if ((i = NS_parse_arg((char *)"-l0", argc, argv, 0)) > 0 || (i = NS_parse_arg((char *)"-logbefore", argc, argv, 0)) > 0)
		takeLogBefore = TRUE;

	if ((i = NS_parse_arg((char *)"-s0", argc, argv, 0)) > 0 || (i = NS_parse_arg((char *)"-sqrtbefore", argc, argv, 0)) > 0)
		takeSqrtBefore = TRUE;

	if ((i = NS_parse_arg((char *)"-n", argc, argv, 1)) > 0 || (i = NS_parse_arg((char *)"-negative", argc, argv, 1)) > 0)
		neg_ratio = atof(argv[i + 1]);

	if ((i = NS_parse_arg((char *)"-a", argc, argv, 1)) > 0 || (i = NS_parse_arg((char *)"-alpha", argc, argv, 1)) > 0)
		alpha = atof(argv[i + 1]);

	if ((i = NS_parse_arg((char *)"-tr", argc, argv, 1)) > 0 || (i = NS_parse_arg((char *)"-truncate", argc, argv, 1)) > 0)
		truncate = atof(argv[i + 1]);

	if ((i = NS_parse_arg((char *)"-ps2", argc, argv, 0)) > 0 || (i = NS_parse_arg((char *)"-contextsmooth2", argc, argv, 0)) > 0)
		contextDistSmooth = 2;
	else if ((i = NS_parse_arg((char *)"-ps", argc, argv, 0)) > 0 || (i = NS_parse_arg((char *)"-contextsmooth", argc, argv, 0)) > 0)
		contextDistSmooth = 1;

	if ((i = NS_parse_arg((char *)"-p", argc, argv, 0)) > 0 || (i = NS_parse_arg((char *)"-pmi", argc, argv, 0)) > 0 ||
		(i = NS_parse_arg((char *)"-p10", argc, argv, 0)) > 0 || (i = NS_parse_arg((char *)"-pmi10", argc, argv, 0)) > 0)
		takePMI = 10;
	else if ((i = NS_parse_arg((char *)"-p2", argc, argv, 0)) > 0 || (i = NS_parse_arg((char *)"-pmi2", argc, argv, 0)) > 0)
		takePMI = 2;

	if ((i = NS_parse_arg((char *)"-cp", argc, argv, 1)) > 0 || (i = NS_parse_arg((char *)"-pmicutoff", argc, argv, 1)) > 0)
		pmiThreshold = atof(argv[i + 1]);

	if ((i = NS_parse_arg((char *)"-sh", argc, argv, 1)) > 0 || (i = NS_parse_arg((char *)"-shift", argc, argv, 1)) > 0)
		pmiShift = atof(argv[i + 1]);

	if ((i = NS_parse_arg((char *)"-v", argc, argv, 1)) > 0 || (i = NS_parse_arg((char *)"-verbose", argc, argv, 1)) > 0)
		verbose = atoi(argv[i + 1]);
	SVDVerbosity = verbose;

	if ((i = NS_parse_arg((char *)"-h", argc, argv, 0)) > 0 || (i = NS_parse_arg((char *)"-help", argc, argv, 0)) > 0)
		NS_usage(argv[0]);

	if ((i = NS_parse_arg((char *)"-e", argc, argv, 1)) > 0 || (i = NS_parse_arg((char *)"-ebound", argc, argv, 1)) > 0)
	{
		las2end[1] = atof(argv[i + 1]);
		las2end[0] = -las2end[1];
	}

	if ((i = NS_parse_arg((char *)"-k", argc, argv, 1)) > 0 || (i = NS_parse_arg((char *)"-kappa", argc, argv, 1)) > 0)
		kappa = atof(argv[i + 1]);

	if ((i = NS_parse_arg((char *)"-w", argc, argv, 1)) > 0 || (i = NS_parse_arg((char *)"-write", argc, argv, 1)) > 0)
		writeMode = atoi(argv[i + 1]);

	omp_set_num_threads(no_thread);
#pragma omp parallel default(none) shared(no_thread)
	{
#pragma omp master
		no_thread = omp_get_num_threads();
	}

	A = NS_loadPrepareMatrix(inputFile, takeLogBefore, takeSqrtBefore, takePMI, pmiThreshold, pmiShift, contextDistSmooth, takeLog, takeSqrt, neg_ratio, alpha, truncate, verbose);
	if (!A) fatalError("failed to read sparse matrix.  Did you specify the correct file type with the -r argument?");

	if (dimensions <= 0) dimensions = imin(A->rows, A->cols);

	if (verbose > 0) printf("Computing the SVD...\n");
	if (!(R = svdLAS2(A, dimensions, iterations, las2end, kappa)))
		fatalError("error in svdLAS2");

	int ir, ic;
	DMat R2 = svdNewDMat(R->Ut->cols, R->d);
	for (ir = 0; ir < R->Ut->cols; ir++)
		for (ic = 0; ic < R->d; ic++)
			R2->value[ir][ic] = (R->Ut->value[ic][ir]) * sqrt(R->S[ic]);

	//NS_normalize_cols(R2);
	//NS_normalize_rows(R2);

	if (vectorFile) {
		printf("Writing embedding file...\n");
		NS_write_output_file(vocabFile, vectorFile, R2);

		//int j;
		//double len = 0.0;
		//printf("\n\nLength of U x sqrt(S)\n");
		//for (i = 0; i < 5; i++)
		//{
		//	len = 0.0;
		//	for (j = 0; j < R2->rows; j++)
		//		len += R2->value[j][i] * R2->value[j][i];
		//	if (len > 0)
		//		len = sqrt(len);
		//	printf("len(U_%d) = %f    ", i, len);
		//}

		if (writeMode >= 2)
		{
			DMat R_temp = svdNewDMat(R->Ut->cols, R->d);
			for (ir = 0; ir < R->Ut->cols; ir++)
				for (ic = 0; ic < R->d; ic++)
					R_temp->value[ir][ic] = (R->Ut->value[ic][ir]) * (R->S[ic]);

			//printf("\n\nLength of U x S\n");
			//for (i = 0; i < 5; i++)
			//{
			//	len = 0.0;
			//	for (j = 0; j < R_temp->rows; j++)
			//		len += R_temp->value[j][i] * R_temp->value[j][i];
			//	if (len > 0)
			//		len = sqrt(len);
			//	printf("len(U_%d) = %f    ", i, len);
			//}

			char filename[512];
			sprintf(filename, "%s-UxS.txt", vectorFile);
			NS_write_output_file(vocabFile, filename, R_temp);
			svdFreeDMat(R_temp);
		}

		if (writeMode >= 3)
		{
			DMat R_temp = svdNewDMat(R->Ut->cols, R->d);
			for (ir = 0; ir < R->Ut->cols; ir++)
				for (ic = 0; ic < R->d; ic++)
					R_temp->value[ir][ic] = (R->Ut->value[ic][ir]);

			//printf("\n\nLength of U\n");
			//for (i = 0; i < 5; i++)
			//{
			//	len = 0.0;
			//	for (j = 0; j < R_temp->rows; j++)
			//		len += R_temp->value[j][i] * R_temp->value[j][i];
			//	if (len > 0)
			//		len = sqrt(len);
			//	printf("len(U_%d) = %f    ", i, len);
			//}

			char filename[512];
			sprintf(filename, "%s-U.txt", vectorFile);
			NS_write_output_file(vocabFile, filename, R_temp);
			svdFreeDMat(R_temp);
		}
	}
	printf("All done!\n");

	free(inputFile); inputFile = NULL;
	free(vocabFile); vocabFile = NULL;
	free(vectorFile); vectorFile = NULL;

	svdFreeSVDRec(R);
	svdFreeDMat(R2);
	svdFreeSMat(A);
	return EXIT_SUCCESS;
}
