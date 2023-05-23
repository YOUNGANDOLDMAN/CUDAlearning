
#include <stdio.h>
#include <cuda.h>

#include "aux.h"

typedef float FLOAT;

/* host, add */
void vec_add_host(FLOAT *x, FLOAT *y, FLOAT *z, int N);

/* device function */
__global__ void vec_add(FLOAT *x, FLOAT *y, FLOAT *z, int N)//和c语言一样的封装函数
{
    /* 1D block */
    int idx = get_tid();//得到线程的全局编号
    if (idx < N) z[idx] = z[idx] + y[idx] + x[idx];//函数运算
}

void vec_add_host(FLOAT *x, FLOAT *y, FLOAT *z, int N)//有意使用c语言，对比纯c语言和使用cuda的时间对比
{
    int i;

    for (i = 0; i < N; i++) z[i] = z[i] + y[i] + x[i];
}

int main()
{
    int N = 20000000;//确定两千万个浮点数
    int nbytes = N * sizeof(FLOAT);//定义两千万个浮点数数组所需要的内存空间

    /* 1D block */
    int bs = 256;//一个block里面有256个线程，还是一维

    /* 2D grid */
    int s = ceil(sqrt((N + bs - 1.) / bs));
    dim3 grid = dim3(s, s);

    FLOAT *dx = NULL, *hx = NULL;
    FLOAT *dy = NULL, *hy = NULL;
    FLOAT *dz = NULL, *hz = NULL;

    int itr = 30;
    int i;
    double th, td;

    /* allocate GPU mem */
    cudaMalloc((void **)&dx, nbytes);
    cudaMalloc((void **)&dy, nbytes);
    cudaMalloc((void **)&dz, nbytes);

    if (dx == NULL || dy == NULL || dz == NULL) {
        printf("couldn't allocate GPU memory\n");
        return -1;
    }

    printf("allocated %.2f MB on GPU\n", nbytes / (1024.f * 1024.f));

    /* alllocate CPU mem */
    hx = (FLOAT *) malloc(nbytes);
    hy = (FLOAT *) malloc(nbytes);
    hz = (FLOAT *) malloc(nbytes);

    if (hx == NULL || hy == NULL || hz == NULL) {
        printf("couldn't allocate CPU memory\n");
        return -2;
    }
    printf("allocated %.2f MB on CPU\n", nbytes / (1024.f * 1024.f));

    /* init */
    for (i = 0; i < N; i++) {
        hx[i] = 1;
        hy[i] = 1;
        hz[i] = 1;
    }

    /* copy data to GPU */
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dz, hz, nbytes, cudaMemcpyHostToDevice);

    /* call GPU */
    cudaDeviceSynchronize();
    td = get_time();
    
    for (i = 0; i < itr; i++) vec_add<<<grid, bs>>>(dx, dy, dz, N);

    cudaDeviceSynchronize();
    td = get_time() - td;

    /* CPU */
    th = get_time();
    for (i = 0; i < itr; i++) vec_add_host(hx, hy, hz, N);
    th = get_time() - th;

    printf("GPU time: %e, CPU time: %e, speedup: %g\n", td, th, th / td);

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    free(hx);
    free(hy);
    free(hz);

    return 0;
}
