
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
    int s = ceil(sqrt((N + bs - 1.) / bs));//确定block的个数
    dim3 grid = dim3(s, s);//确定block的形势是s*s的

    FLOAT *dx = NULL, *hx = NULL;//定义host和device的内存
    FLOAT *dy = NULL, *hy = NULL;
    FLOAT *dz = NULL, *hz = NULL;

    int itr = 30;
    int i;
    double th, td;

    /* allocate GPU mem */
    cudaMalloc((void **)&dx, nbytes);//申请内存空间，这就是申请空间的格式
    cudaMalloc((void **)&dy, nbytes);//dy，dx，dz是申请空间的向量，nbytes是申请的大小
    cudaMalloc((void **)&dz, nbytes);

    if (dx == NULL || dy == NULL || dz == NULL) {//判断是否为空，确定是否运行
        printf("couldn't allocate GPU memory\n");
        return -1;
    }

    printf("allocated %.2f MB on GPU\n", nbytes / (1024.f * 1024.f));

    /* alllocate CPU mem */
    hx = (FLOAT *) malloc(nbytes);//标准c语言的函数，申请内存
    hy = (FLOAT *) malloc(nbytes);
    hz = (FLOAT *) malloc(nbytes);

    if (hx == NULL || hy == NULL || hz == NULL) {
        printf("couldn't allocate CPU memory\n");
        return -2;
    }
    printf("allocated %.2f MB on CPU\n", nbytes / (1024.f * 1024.f));

    /* init */
    for (i = 0; i < N; i++) {
        hx[i] = 1;//对其进行初始化
        hy[i] = 1;
        hz[i] = 1;
    }

    /* copy data to GPU */
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);//（目的地，sourse来源，内存大小，方向host——device）
    cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dz, hz, nbytes, cudaMemcpyHostToDevice);

    /* call GPU */
    cudaDeviceSynchronize();//等GPU计算完CPU再运行的函数
    td = get_time();//获取时间
    
    for (i = 0; i < itr; i++) vec_add<<<grid, bs>>>(dx, dy, dz, N);//运行cuda的函数

    cudaDeviceSynchronize();//等GPU计算完CPU再运行的函数
    td = get_time() - td;//计算时间

    /* CPU */
    th = get_time();
    for (i = 0; i < itr; i++) vec_add_host(hx, hy, hz, N);//运行普通c语言的函数
    th = get_time() - th;//计算时间

    printf("GPU time: %e, CPU time: %e, speedup: %g\n", td, th, th / td);

    cudaFree(dx);//申请的内存要释放掉
    cudaFree(dy);
    cudaFree(dz);

    free(hx);
    free(hy);
    free(hz);

    return 0;
}
