
#include <stdio.h>
#include <cuda.h>

typedef double FLOAT;//定义FLOAT为双精度的

__global__ void sum(FLOAT *x)
{
    int tid = threadIdx.x;//获取线程索引，一维数组

    x[tid] += 1;//每个数+1
    
}

int main()
{
    int N = 32;//定义数组的元素数目
    int nbytes = N * sizeof(FLOAT);//确定数组的大小，方便下面申请内存

    FLOAT *dx = NULL, *hx = NULL;//初始化gpu和cpu的内存大小
    int i;

    /* allocate GPU mem */
    cudaMalloc((void **)&dx, nbytes);//申请内存

    if (dx == NULL) {
        printf("couldn't allocate GPU memory\n");
        return -1;
    }

    /* alllocate CPU host mem: memory copy is faster than malloc */
    hx = (FLOAT *)malloc(nbytes);

    if (hx == NULL) {
        printf("couldn't allocate CPU memory\n");
        return -2;
    }

    /* init */
    printf("hx original: \n");
    for (i = 0; i < N; i++) {
        hx[i] = i;

        printf("%g\n", hx[i]);
    }

    /* copy data to GPU */
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);

    /* call GPU */
    sum<<<1, N>>>(dx);

    /* let GPU finish */
    cudaDeviceSynchronize();

    /* copy data from GPU */
    cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);

    printf("\nhx from GPU: \n");
    for (i = 0; i < N; i++) {
        printf("%g\n", hx[i]);
    }

    cudaFree(dx);
    free(hx);

    return 0;
}
