#include "vhf/fblas.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include <stdlib.h>

int get_omp_threads();
int omp_get_thread_num();

#include "fftw3.h"

#define BUNCHSIZE 1024

// #define EXTRA_ALLOC 63
// void *align_to_512bit(void *ptr)
// {
//     uintptr_t addr = (uintptr_t)ptr;
//     return (void *)((addr + 63) & ~63);
// }

void _FFT_Matrix_Col_InPlace(double *matrix, // the size of matrix should be (nRow, nCol* *mesh)
                             int nRow, int nCol, int *mesh,
                             double *buf)
{
    int mesh_complex[3] = {mesh[0], mesh[1], mesh[2] / 2 + 1};
    int64_t nComplex = mesh_complex[0] * mesh_complex[1] * mesh_complex[2];
    int64_t nReal = mesh[0] * mesh[1] * mesh[2];
    const int nThread = get_omp_threads();

    const int64_t m = nRow;
    const int64_t n = nCol * mesh[0] * mesh[1] * mesh[2];
    const int64_t n_complex = nCol * mesh_complex[0] * mesh_complex[1] * mesh_complex[2];
    const int64_t nMesh = mesh[0] * mesh[1] * mesh[2];
    const int64_t nMeshComplex = mesh_complex[0] * mesh_complex[1] * mesh_complex[2];

    // (1) transform (Row, Block, Col) -> (Row, Col, Block)

#pragma omp parallel for num_threads(nThread)
    for (int64_t i = 0; i < m; i++)
    {
        int64_t iCol = 0;

        for (int64_t iBlock = 0; iBlock < nMesh; iBlock++)
        {
            for (int64_t j = 0; j < nCol; j++, iCol++)
            {
                buf[i * n + j * nMesh + iBlock] = matrix[i * n + iCol];
            }
        }
    }

    // (2) perform FFT on the last dimension

    int64_t nFFT = nRow * nCol;

    double __complex__ *mat_complex = (double __complex__ *)buf;
    double __complex__ *buf_complex = (double __complex__ *)matrix;

    // create plan

    const int BunchSize = nFFT / nThread + 1;

#pragma omp parallel num_threads(nThread)
    {
        int tid = omp_get_thread_num();
        int64_t start = tid * BunchSize;
        int64_t end = (tid + 1) * BunchSize;
        if (end > nFFT)
        {
            end = nFFT;
        }

        //// allocate buffer to fix align issues

        double *local_buf_real = fftw_alloc_real(BUNCHSIZE * nReal);
        fftw_complex *local_buf_complex = fftw_alloc_complex(BUNCHSIZE * nComplex);

        int nBunch = (end - start) / BUNCHSIZE;
        int nleft = (end - start) % BUNCHSIZE;

        if (nBunch)
        {
            fftw_plan plan = fftw_plan_many_dft_r2c(3, mesh, BUNCHSIZE, local_buf_real, mesh, 1, nReal, local_buf_complex, mesh_complex, 1, nComplex, FFTW_ESTIMATE);
            for (int i = 0; i < nBunch; i++)
            {
                memcpy(local_buf_real, buf + (start + i * BUNCHSIZE) * nReal, sizeof(double) * BUNCHSIZE * nReal);
                fftw_execute_dft_r2c(plan, local_buf_real, local_buf_complex);
                memcpy(buf_complex + (start + i * BUNCHSIZE) * nComplex, local_buf_complex, sizeof(double __complex__) * BUNCHSIZE * nComplex);
            }
            fftw_destroy_plan(plan);
        }

        if (nleft)
        {
            fftw_plan plan = fftw_plan_many_dft_r2c(3, mesh, nleft, local_buf_real, mesh, 1, nReal, local_buf_complex, mesh_complex, 1, nComplex, FFTW_ESTIMATE);
            memcpy(local_buf_real, buf + (start + nBunch * BUNCHSIZE) * nReal, sizeof(double) * nleft * nReal);
            fftw_execute_dft_r2c(plan, local_buf_real, local_buf_complex);
            memcpy(buf_complex + (start + nBunch * BUNCHSIZE) * nComplex, local_buf_complex, sizeof(double __complex__) * nleft * nComplex);
            fftw_destroy_plan(plan);
        }

        // fftw_plan plan = fftw_plan_many_dft_r2c(3, mesh, end - start, buf + start * nReal, mesh, 1, nReal, (fftw_complex *)buf_complex + start * nComplex, mesh_complex, 1, nComplex, FFTW_ESTIMATE);
        // fftw_execute(plan);
        // fftw_destroy_plan(plan);

        fftw_free(local_buf_real);
        fftw_free(local_buf_complex);
    }

    // (3) transform (Row, Col, Block) -> (Row, Block, Col)

    mat_complex = (double __complex__ *)matrix;
    buf_complex = (double __complex__ *)buf;

#pragma omp parallel for num_threads(nThread)
    for (int64_t i = 0; i < m; i++)
    {
        int64_t iCol = 0;

        for (int64_t j = 0; j < nCol; j++)
        {
            for (int64_t iBlock = 0; iBlock < nMeshComplex; iBlock++, iCol++)
            {
                buf_complex[i * n_complex + iBlock * nCol + j] = mat_complex[i * n_complex + iCol];
            }
        }
    }

    memcpy(matrix, buf, sizeof(double __complex__) * m * nCol * mesh_complex[0] * mesh_complex[1] * mesh_complex[2]);
}

void _iFFT_Matrix_Col_InPlace(double __complex__ *matrix, // the size of matrix should be (nRow, nCol* *mesh)
                              int nRow, int nCol, int *mesh,
                              double __complex__ *buf)
{
    int mesh_complex[3] = {mesh[0], mesh[1], mesh[2] / 2 + 1};
    int64_t nComplex = mesh_complex[0] * mesh_complex[1] * mesh_complex[2];
    int64_t nReal = mesh[0] * mesh[1] * mesh[2];
    const int64_t nThread = get_omp_threads();

    const int64_t m = nRow;
    const int64_t n = nCol * mesh[0] * mesh[1] * mesh[2];
    const int64_t n_Complex = nCol * mesh_complex[0] * mesh_complex[1] * mesh_complex[2];
    const int64_t nMesh = mesh[0] * mesh[1] * mesh[2];
    const int64_t nMeshComplex = mesh_complex[0] * mesh_complex[1] * mesh_complex[2];
    const double factor = 1.0 / (double)(nMesh);

    // (1) transform (Row, Block, Col) -> (Row, Col, Block)

#pragma omp parallel for num_threads(nThread)
    for (int64_t i = 0; i < m; i++)
    {
        int64_t iCol = 0;

        for (int64_t iBlock = 0; iBlock < nMeshComplex; iBlock++)
        {
            for (int64_t j = 0; j < nCol; j++, iCol++)
            {
                buf[i * n_Complex + j * nMeshComplex + iBlock] = matrix[i * n_Complex + iCol];
            }
        }
    }

    // (2) perform iFFT on the last dimension

    int64_t nFFT = nRow * nCol;

    double *mat_real = (double *)buf;
    double *buf_real = (double *)matrix;

    // create plan

    const int64_t BunchSize = nFFT / nThread + 1;

#pragma omp parallel num_threads(nThread)
    {
        int64_t tid = omp_get_thread_num();
        int64_t start = tid * BunchSize;
        int64_t end = (tid + 1) * BunchSize;
        if (end > nFFT)
        {
            end = nFFT;
        }

        // fftw_plan plan = fftw_plan_many_dft_c2r(3, mesh, end - start, (fftw_complex *)buf + start * nComplex, mesh_complex, 1, nComplex, buf_real + start * nReal, mesh, 1, nReal, FFTW_ESTIMATE);
        // fftw_execute(plan);
        // fftw_destroy_plan(plan);

        /// allocate buffer to fix align issues

        fftw_complex *local_buf_complex = fftw_alloc_complex(BUNCHSIZE * nComplex);
        double *local_buf_real = fftw_alloc_real(BUNCHSIZE * nReal);
        int nBunch = (end - start) / BUNCHSIZE;
        int nleft = (end - start) % BUNCHSIZE;
        if (nBunch)
        {
            fftw_plan plan = fftw_plan_many_dft_c2r(3, mesh, BUNCHSIZE, local_buf_complex, mesh_complex, 1, nComplex, local_buf_real, mesh, 1, nReal, FFTW_ESTIMATE);
            for (int i = 0; i < nBunch; i++)
            {
                memcpy(local_buf_complex, (fftw_complex *)buf + (start + i * BUNCHSIZE) * nComplex, sizeof(double __complex__) * BUNCHSIZE * nComplex);
                fftw_execute_dft_c2r(plan, local_buf_complex, local_buf_real);
                memcpy(buf_real + (start + i * BUNCHSIZE) * nReal, local_buf_real, sizeof(double) * BUNCHSIZE * nReal);
            }
            fftw_destroy_plan(plan);
        }
        if (nleft)
        {
            fftw_plan plan = fftw_plan_many_dft_c2r(3, mesh, nleft, local_buf_complex, mesh_complex, 1, nComplex, local_buf_real, mesh, 1, nReal, FFTW_ESTIMATE);
            memcpy(local_buf_complex, (fftw_complex *)buf + (start + nBunch * BUNCHSIZE) * nComplex, sizeof(double __complex__) * nleft * nComplex);
            fftw_execute_dft_c2r(plan, local_buf_complex, local_buf_real);
            memcpy(buf_real + (start + nBunch * BUNCHSIZE) * nReal, local_buf_real, sizeof(double) * nleft * nReal);
            fftw_destroy_plan(plan);
        }
        fftw_free(local_buf_real);
        fftw_free(local_buf_complex);
    }

    // (3) transform (Row, Col, Block) -> (Row, Block, Col)

    mat_real = (double *)matrix;
    buf_real = (double *)buf;

#pragma omp parallel for num_threads(nThread)
    for (int64_t i = 0; i < m; i++)
    {
        int64_t iCol = 0;

        for (int64_t j = 0; j < nCol; j++)
        {
            for (int64_t iBlock = 0; iBlock < nMesh; iBlock++, iCol++)
            {
                buf_real[i * n + iBlock * nCol + j] = mat_real[i * n + iCol] * factor;
            }
        }
    }

    memcpy(mat_real, buf_real, sizeof(double) * m * nCol * mesh[0] * mesh[1] * mesh[2]);
}