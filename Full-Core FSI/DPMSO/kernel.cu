
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream> 
#include <stdlib.h>
#include <cusparse.h>
#include<cstring>
#include<iostream>
#include <float.h>
#include <sys/time.h>
#include <nmmintrin.h>

#include "mtx.hpp"
#include "class.hpp"
#include"scoo.hpp"
#include"cpu_spmv.hpp"
#include"config.hpp"
#include"csr_cusparse.hpp"

#define CHECK_CUDA(func) { \
    cudaError_t err = (func); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

using namespace std;

const int num_elements = 10240;

// GPU Kernel Function
__global__ void occa_ellipticPartialAxCoeffHex3D_v1_0(
    float *D, float *ggeo, float *lambda0, float *lambda1, float *q, float *Aq)
{
    int e = 0 + blockIdx.x;
    __shared__ float s_D[8][8];
    __shared__ float s_q[8][8];
    __shared__ float s_v[8][8];
    __shared__ float s_w[8][8];
    float r_GDut, r_Auk;

    // register array to hold u(i,j,0:N) private to thread
    float r_u[8];
    // array for results Au(i,j,0:N)
    float r_Au[8];
    int element = e;

    int j = 0 + threadIdx.y;
    int i = 0 + threadIdx.x;

    // Load D into shared memory
    s_D[j][i] = D[8 * j + i];

    const int base = i + j * 8 + element * 512;

    // Load pencil of u into register
#pragma unroll 8
    for (int k = 0; k < 8; k++)
    {
        const int id = base + k * 8 * 8;
        r_u[k] = q[id];
    }
#pragma unroll 8
    for (int k = 0; k < 8; k++)
    {
        r_Au[k] = 0.0;
    }

    // Layer by layer
    for (int k = 0; k < 8; k++)
    {
        __syncthreads();
        s_q[j][i] = r_u[k];

        __syncthreads();

        const int gbase = element * 7 * 512 + k * 8 * 8 + j * 8 + i;
        const float r_G00 = ggeo[gbase + 0 * 512];
        const float r_G01 = ggeo[gbase + 1 * 512];
        const float r_G02 = ggeo[gbase + 4 * 512];
        const float r_G11 = ggeo[gbase + 2 * 512];
        const float r_G12 = ggeo[gbase + 3 * 512];
        const float r_G22 = ggeo[gbase + 5 * 512];

        float ur = 0.f, us = 0.f, ut = 0;
#pragma unroll 8
        for (int m = 0; m < 8; m++)
        {
            ut += s_D[k][m] * r_u[m]; //
        }
#pragma unroll 8
        for (int m = 0; m < 8; m++)
        {
            ur += s_D[i][m] * s_q[j][m];
            us += s_D[j][m] * s_q[m][i];
        }

        const int id = element * 512 + k * 8 * 8 + j * 8 + i;
        const float lbda0 = lambda0[id];
        s_w[j][i] = lbda0 * (r_G01 * ur + r_G11 * us + r_G12 * ut);
        s_v[j][i] = lbda0 * (r_G00 * ur + r_G01 * us + r_G02 * ut);
        r_GDut = lbda0 * (r_G02 * ur + r_G12 * us + r_G22 * ut);
        r_Auk = 0.0;

        __syncthreads();

#pragma unroll 8
        for (int m = 0; m < 8; m++)
        {
            r_Au[m] += s_D[k][m] * r_GDut; //
        }
#pragma unroll 8
        for (int m = 0; m < 8; m++)
        {
            r_Auk += s_D[m][j] * s_w[m][i];
            r_Auk += s_D[m][i] * s_v[j][m]; // 寄存器复用（sm
        }
        r_Au[k] += r_Auk;
    }

    // Write out results
    __syncthreads();
#pragma unroll 8
    for (int k = 0; k < 8; k++)
    {
        Aq[base + k * 8 * 8] = r_Au[k];
    }
}

// Host-side function for initializing arrays
void initialize_arrays(float *D, float *ggeo, float *lambda0, float *lambda1, float *q, float *Aq)
{
    for (int i = 0; i < 64; i++) 
    {
        D[i] = (float)rand() / RAND_MAX;
    }
    for (int element = 0; element < num_elements; element++) {
        for (int i = 0; i < 512 * 7; i++) 
        {
            int id = element * 512 * 7 + i;
            ggeo[id] = (float)rand() / RAND_MAX;
        }
        for (int i = 0; i < 512; i++) 
        {
            int id = element * 512 + i;
            lambda0[id] = lambda1[i] = (float)rand() / RAND_MAX;
        }
        for (int i = 0; i < 512; i++)
        {
            int id = element * 512 + i;
            q[id] = (float)rand() / RAND_MAX;
            Aq[id] = 0.0;
        }
    }
    
}

#pragma warning( disable : 4996)
#define _CRT_SECURE_NO_WARNINGS

inline int DIV(int x, int tile_size) {
	if (x % tile_size) return x / tile_size + 1; else return x / tile_size;
}

template<class dataType>
void scoo_spmv(SCOO<dataType>& sc, dataType* x_vector, dataType* value_res);

//test kernel 
__global__ void testkernel() {
	int warpid = threadIdx.x;
	int lid = threadIdx.y * blockDim.x + threadIdx.x;
	//int gid = blockIdx.x * blockDim.x * blockDim.y + lid;
	int orderid = blockDim.y * blockIdx.x + threadIdx.y;
	printf("warpid=%d\tlid=%d\tgid=%d\n", warpid, lid, orderid);
}
//kernel
template<class dataType>
__global__ void _spmv_cpart1(int cluster_ordernum, unsigned int* order, unsigned int* order_accumulate_array,
	unsigned int* row_index_array, dataType* data, dataType* x_vector, dataType* value_res) {
	int orderid = blockDim.y * blockIdx.x + threadIdx.y;
	if (orderid < cluster_ordernum) {
		__shared__ dataType vector_buf[BLOCK_DIM_1];
		__shared__ unsigned int ordernum_buf[BLOCK_DIM_1 + 1];
		//here can be tested by if(threadIdx.x == 0)
		//when the block num is bigger than 32 it can be reused
		if (threadIdx.x < 2) {
			ordernum_buf[threadIdx.y + threadIdx.x] = order_accumulate_array[orderid + threadIdx.x];
			//vector连续
			vector_buf[threadIdx.y] = x_vector[order[orderid]];
			//printf("ordernum_buf[threadIdx.y] = %d, ordernum_buf[threadIdx.y + 1] = %d\n",ordernum_buf[threadIdx.y],ordernum_buf[threadIdx.y + 1]);
			//printf("vector_buf[threadIdx.y] = %d", vector_buf[threadIdx.y]);
		}

		// 32 is the number of the thread in the warp
		for (int i = ordernum_buf[threadIdx.y]; i < ordernum_buf[threadIdx.y + 1]; i = i + 32) {
			if (threadIdx.x < ordernum_buf[threadIdx.y + 1] - i) {
				int rownum = row_index_array[threadIdx.x + i];
				dataType res = data[threadIdx.x + i] * vector_buf[threadIdx.y];
				//printf("threadIdx.x = %d,orderid = %d,cpart1_res1 = %lf\n",threadIdx.x,orderid,res);
				atomicAdd(&value_res[rownum], res);
			}
		}
	}
}

template<class dataType>
__global__ void _spmv_cpart1_dewarp(int cluster_ordernum, unsigned int* order, unsigned int* order_accumulate_array,
	unsigned int* row_index_array, dataType* data, dataType* x_vector, dataType* value_res) {
	int warporder = (blockDim.y * blockIdx.x + threadIdx.y) * KERNEL1_PART;
	if (warporder < cluster_ordernum) {
		__shared__ dataType vector_buf[BLOCK_DIM_1 * KERNEL1_PART];
		__shared__ unsigned int ordernum_buf[BLOCK_DIM_1 * KERNEL1_PART + 1];
		//here can be tested by if(threadIdx.x == 0)
		//when the block num is bigger than 32 it can be reused
		/*
		//all threads read data
		int lid = threadIdx.y * blockDim.x + threadIdx.x;
		if (lid < BLOCK_DIM_1 * KERNEL1_PART) {
			//vector连续
			vector_buf[lid] = x_vector[order[warporder + lid]];
		}
		if (lid < BLOCK_DIM_1 * KERNEL1_PART + 1) {
			ordernum_buf[lid] = order_accumulate_array[warporder + lid];
		}
		__syncthreads();
		*/
		if (threadIdx.x < KERNEL1_PART) {
			vector_buf[threadIdx.y * KERNEL1_PART + threadIdx.x] = x_vector[order[warporder + threadIdx.x]];
		}
		if (threadIdx.x < KERNEL1_PART + 1) {
			ordernum_buf[threadIdx.y * KERNEL1_PART + threadIdx.x] = order_accumulate_array[warporder + threadIdx.x];
		}
		int orderid = threadIdx.y * KERNEL1_PART + threadIdx.x / (32 / KERNEL1_PART);
		int idx = threadIdx.x % (32 / KERNEL1_PART);
		//printf("orderid = %d\n", orderid);
		// 32 is the number of the thread in the warp
		for (int i = ordernum_buf[orderid]; i < ordernum_buf[orderid + 1]; i = i + (32 / KERNEL1_PART)) {
			//printf("ordernum_buf[orderid] = %d, ordernum_buf[orderid + 1] = %d\n", ordernum_buf[orderid], ordernum_buf[orderid + 1]);
			if (idx < ordernum_buf[orderid + 1] - i) {
				int rownum = row_index_array[idx + i];
				//if(warporder + threadIdx.x / (32 / KERNEL1_PART) == 255)
				//printf("threadIdx.x / (32 / KERNEL1_PART) = %d,warporder = %d,rownum = %d\n", threadIdx.x / (32 / KERNEL1_PART), warporder, rownum);
				dataType res = data[idx + i] * vector_buf[orderid];
				//printf("threadIdx.x = %d,orderid = %d,cpart1_res1 = %lf\n",threadIdx.x,orderid,res);
				atomicAdd(&value_res[rownum], res);
			}
		}

	}
}

//SpMV kernel one, order saved by compressed format
template<class dataType>
__global__ void _spmv_cpart1_bit(int cluster_blocknum, unsigned int* order, unsigned int* order_res_entry, unsigned int* order_bitmap,
	unsigned int* row_index_array, dataType* data, dataType* x_vector, dataType* value_res) {
	int lid = threadIdx.y * blockDim.x + threadIdx.x;
	int orderid = blockIdx.x * blockDim.x * blockDim.y + lid;
	if (orderid < cluster_blocknum) {
		__shared__ dataType vector_buf[BLOCK_DIM_1 * 32];
		__shared__ unsigned int basement;
		__shared__ unsigned int vectornum;
		__shared__ unsigned int warp_base[BLOCK_DIM_1];
		__shared__ unsigned int bitmap_buf[BLOCK_DIM_1];
		if (lid < 1) {
			basement = order_res_entry[blockIdx.x];
			vectornum = order_res_entry[blockIdx.x + 1];
			//printf("blockIdx.x = %d,gid = %d,vectornum = %d\n", blockIdx.x, orderid, vectornum);
		}
		if (lid < BLOCK_DIM_1) {
			bitmap_buf[lid] = order_bitmap[lid + blockIdx.x * blockDim.y];
			warp_base[lid] = __popc(bitmap_buf[lid]);
			//printf("lid = %d ,gid = %d,warp_size = %d\n",lid, orderid, warp_base[lid]);
		}
		__syncthreads();
		if (lid < vectornum - basement + 1) {
			vector_buf[lid] = x_vector[order[basement + lid]];
			//printf("blockIdx.x = %d,lid = %d ,gid = %d,order = %d, x_vector= %lf\n",blockIdx.x, lid, orderid, order[basement + lid], vector_buf[lid]);
		}
		__syncthreads();

		unsigned int mask = 0xFFFFFFFF << (32 - threadIdx.x);
		unsigned int rownum = 0;
		for (int j = 0; j < threadIdx.y; j++) {
			rownum += warp_base[j];
		}
		rownum += __popc(bitmap_buf[threadIdx.y] & mask);
		//printf("blockIdx.x = %d,lid = %d ,gid = %d,rownum = %d, data= %lf\n", blockIdx.x, lid, orderid, rownum, vector_buf[rownum]);
		dataType res = data[orderid] * vector_buf[rownum];
		//printf("blockIdx.x = %d,lid = %d ,gid = %d,rownum = %d, data= %lf\n", blockIdx.x, lid, orderid, rownum, data[orderid]);
		atomicAdd(&value_res[row_index_array[orderid]], res);

	}
}


template<class dataType>
__global__ void _spmv_cpart2(int cluster_ordernum, unsigned int* order_accumulate_array, unsigned int* row_index_array,
	unsigned int* col_index_array, dataType* data, dataType* x_vector, dataType* value_res) {
	//thread local id
	int orderid = blockDim.y * blockIdx.x + threadIdx.y;
	if (orderid < cluster_ordernum) {
		__shared__ dataType vector_buf[BLOCK_DIM_2 * 2];
		__shared__ unsigned int ordernum_buf[BLOCK_DIM_2 + 1];
		//here can be tested by if(threadIdx.x == 0)
		//when the block num is bigger than 32 it can be reused
		if (threadIdx.x < 2) {
			ordernum_buf[threadIdx.y + threadIdx.x] = order_accumulate_array[orderid + threadIdx.x];
			vector_buf[threadIdx.y * 2 + threadIdx.x] = x_vector[col_index_array[orderid * 2 + threadIdx.x]];
		}

		// 32 is the number of the thread in the warp
		for (int i = ordernum_buf[threadIdx.y]; i < ordernum_buf[threadIdx.y + 1]; i = i + 32) {
			if (threadIdx.x < ordernum_buf[threadIdx.y + 1] - i) {
				int rownum = row_index_array[threadIdx.x + i];
				dataType res = data[(threadIdx.x + i) * 2] * vector_buf[threadIdx.y * 2];
				res += data[(threadIdx.x + i) * 2 + 1] * vector_buf[threadIdx.y * 2 + 1];
				atomicAdd(&value_res[rownum], res);
			}
		}
	}
}

template<class dataType>
__global__ void _spmv_cpart2_dewarp(int cluster_ordernum, unsigned int* order_accumulate_array, unsigned int* row_index_array,
	unsigned int* col_index_array, dataType * data, dataType * x_vector, dataType * value_res) {
	//thread local id
	int warporder = (blockDim.y * blockIdx.x + threadIdx.y) * KERNEL2_PART;
	if (warporder < cluster_ordernum) {
		__shared__ dataType vector_buf[BLOCK_DIM_2 * KERNEL2_PART * 2];
		__shared__ unsigned int ordernum_buf[BLOCK_DIM_2 * KERNEL2_PART + 1];
		//here can be tested by if(threadIdx.x == 0)
		//when the block num is bigger than 32 it can be reused
		/*
		//all threads read data
		int lid = threadIdx.y * blockDim.x + threadIdx.x;
		int gid = blockIdx.x * blockDim.x * blockDim.y + lid;

		if (lid < BLOCK_DIM_2 * KERNEL2_PART * 2) {
			//vector连续
			vector_buf[lid] = x_vector[col_index_array[warporder * 2 + lid]];
		}
		if (lid < BLOCK_DIM_2 * KERNEL2_PART + 1) {
			ordernum_buf[lid] = order_accumulate_array[warporder + lid];
			//printf("gid = %d,ordernum = %d\n", gid, ordernum_buf[lid]);
		}
		__syncthreads();
		*/
		if (threadIdx.x < KERNEL2_PART * 2) {
			vector_buf[threadIdx.y * KERNEL2_PART * 2 + threadIdx.x] = x_vector[col_index_array[warporder * 2 + threadIdx.x]];
			//printf("gid = %d,warporder = %d,vector_buf = %lf\n",gid, warporder, vector_buf[threadIdx.y * KERNEL2_PART + threadIdx.x]);
		}
		if (threadIdx.x < KERNEL2_PART + 1) {
			ordernum_buf[threadIdx.y * KERNEL2_PART + threadIdx.x] = order_accumulate_array[warporder + threadIdx.x];
			//printf("ordernum_buf[threadIdx.y * KERNEL2_PART + threadIdx.x]=%d\n", ordernum_buf[threadIdx.y * KERNEL2_PART + threadIdx.x]);

		}

		int orderid = threadIdx.y * KERNEL2_PART + threadIdx.x / (32 / KERNEL2_PART);
		int idx = threadIdx.x % (32 / KERNEL2_PART);
		//printf("orderid = %d\n", orderid);
		// 32 is the number of the thread in the warp
		for (int i = ordernum_buf[orderid]; i < ordernum_buf[orderid + 1]; i = i + (32 / KERNEL2_PART)) {
			//printf("gid = %d, i = %d,ordernum_buf[orderid] = %d, ordernum_buf[orderid + 1] = %d\n", gid, i, ordernum_buf[orderid], ordernum_buf[orderid + 1]);
			if (idx < ordernum_buf[orderid + 1] - i) {
				int rownum = row_index_array[idx + i];
				//printf("id = %d,rownum = %d\n",idx+i, rownum);
				//if(warporder + threadIdx.x / (32 / KERNEL1_PART) == 255)
				//printf("threadIdx.x / (32 / KERNEL1_PART) = %d,warporder = %d,rownum = %d\n", threadIdx.x / (32 / KERNEL1_PART), warporder, rownum);
				dataType res = data[(idx + i) * 2] * vector_buf[orderid * 2];
				res += data[(idx + i) * 2 + 1] * vector_buf[orderid * 2 + 1];
				//printf("threadIdx.x = %d,orderid = %d,cpart1_res1 = %lf\n",threadIdx.x,orderid,res);
				//if(rownum == 66)
				//printf("gid = %d,lid = %d,warporder = %d,res = %lf,rownum = %d\n", blockIdx.x * blockDim.x * blockDim.y + lid,lid,warporder, res, rownum);
				atomicAdd(&value_res[rownum], res);
			}
		}

	}
}

//SpMV kernel one, order saved by compressed format
template<class dataType>
__global__ void _spmv_cpart2_bit(int cluster_blocknum, unsigned int* order_res_entry, unsigned int* order_bitmap,
	unsigned int* row_index_array, unsigned int* col_index_array, dataType * data, dataType * x_vector, dataType * value_res) {
	int lid = threadIdx.y * blockDim.x + threadIdx.x;
	int orderid = blockIdx.x * blockDim.x * blockDim.y + lid;
	if (orderid < cluster_blocknum) {
		__shared__ dataType vector_buf[BLOCK_DIM_2 * 32 * 2];
		__shared__ unsigned int basement;
		__shared__ unsigned int vectornum;
		__shared__ unsigned int warp_base[BLOCK_DIM_2];
		__shared__ unsigned int bitmap_buf[BLOCK_DIM_2];
		if (lid < 1) {
			basement = order_res_entry[blockIdx.x];
			vectornum = order_res_entry[blockIdx.x + 1];
			//printf("blockIdx.x = %d,gid = %d,vectornum = %d\n", blockIdx.x, orderid, vectornum);
		}
		if (lid < BLOCK_DIM_2) {
			bitmap_buf[lid] = order_bitmap[lid + blockIdx.x * blockDim.y];
			warp_base[lid] = __popc(bitmap_buf[lid]);
			//printf("lid = %d ,gid = %d,warp_size = %d\n",lid, orderid, warp_base[lid]);
		}
		__syncthreads();

		if (lid < vectornum - basement + 1) {
			vector_buf[lid * 2] = x_vector[col_index_array[basement * 2 + lid * 2]];
			vector_buf[lid * 2 + 1] = x_vector[col_index_array[basement * 2 + lid * 2 + 1]];
			//printf("blockIdx.x = %d,lid = %d ,gid = %d,order = %d, x_vector= %lf\n",blockIdx.x, lid, orderid, order[basement + lid], vector_buf[lid]);
		}
		/*  There is a bug.
		for (int i = lid; i < (vectornum - basement) * 2 + 2; i += BLOCK_DIM_2 * 32) {
			vector_buf[i] = x_vector[col_index_array[basement * 2 + i]];
			//printf("blockIdx.x = %d,lid = %d ,gid = %d,order = %d, x_vector= %lf\n",blockIdx.x, lid, orderid, order[basement + lid], vector_buf[lid]);
		}
		*/
		__syncthreads();

		unsigned int mask = 0xFFFFFFFF << (32 - threadIdx.x);
		unsigned int rownum = 0;
		for (int j = 0; j < threadIdx.y; j++) {
			rownum += warp_base[j];
		}
		rownum += __popc(bitmap_buf[threadIdx.y] & mask);
		//printf("blockIdx.x = %d,lid = %d ,gid = %d,rownum = %d, data= %lf\n", blockIdx.x, lid, orderid, rownum, vector_buf[rownum]);
		dataType res = data[orderid * 2] * vector_buf[rownum * 2];
		res += data[orderid * 2 + 1] * vector_buf[rownum * 2 + 1];
		//printf("blockIdx.x = %d,lid = %d ,gid = %d,rownum = %d, data= %lf\n", blockIdx.x, lid, orderid, rownum, data[orderid]);
		atomicAdd(&value_res[row_index_array[orderid]], res);

	}
}


template<class dataType>
__global__ void _spmv_discretepart(int discrete_blocknum, unsigned int* index_accumulate_array, unsigned int* row_index_array,
	unsigned int* col_index_array, dataType * data, dataType * x_vector, dataType * value_res) {
	int orderid = blockIdx.x * blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x;
	if (orderid < discrete_blocknum) {
		dataType res = 0;
		for (int i = index_accumulate_array[orderid]; i < index_accumulate_array[orderid + 1]; i++) {
			res += data[i] * x_vector[col_index_array[i]];
		}
		atomicAdd(&value_res[row_index_array[orderid]], res);
	}
}


template<class dataType>
float scoo_spmv(SCOO<dataType> sc, dataType * x_vector, dataType * value_res, Plan best) {
	cudaError_t cudaStatus;

	assignblockdim1(best);
	assignblockdim2(best);
	assignblockdim3(best);
	assignkernel1_th(best);
	assignkernel2_th(best);

	dim3 blocksize1, blocksize2, blocksize3;
	dim3 gridsize1, gridsize2, gridsize3;

	dataType* dev_vector = 0x0;
	dataType* dev_value_res = 0x0;
	//cpart1
	unsigned int* dev_cpart1_order = 0x0;
	unsigned int* dev_cpart1_row = 0x0;
	unsigned int* dev_cpart1_bitmap = 0x0;
	unsigned int* dev_cpart1_resentry = 0x0;
	unsigned int* dev_cpart1_orderacc = 0x0;
	dataType* dev_cpart1_data = 0x0;

	//cpart2
	unsigned int* dev_cpart2_orderacc = 0x0;
	unsigned int* dev_cpart2_row = 0x0;
	unsigned int* dev_cpart2_col = 0x0;
	unsigned int* dev_cpart2_bitmap = 0x0;
	unsigned int* dev_cpart2_resentry = 0x0;
	dataType* dev_cpart2_data = 0x0;

	//discrete part
	unsigned int* dev_discrete_row = 0x0;
	unsigned int* dev_discrete_col = 0x0;
	unsigned int* dev_discrete_index = 0x0;
	dataType* dev_discrete_data = 0x0;

	if (best.kernel1 == 1) {
		cpart1tocompress(&sc.cluster_part1);

		blocksize1.x = 32;
		blocksize1.y = BLOCK_DIM_1;
		gridsize1.x = DIV(sc.cluster_part1.cluster_blocknum, BLOCK_DIM_1 * 32);

		cudaStatus = cudaMalloc((void**)& dev_cpart1_bitmap, (sc.cluster_part1.cluster_bitmapnum) * sizeof(unsigned int));
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n", cudaStatus, cudaGetErrorString(cudaStatus));
			return -1;
		}
		cudaStatus = cudaMalloc((void**)& dev_cpart1_resentry, (sc.cluster_part1.cluster_resnum) * sizeof(unsigned int));
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n", cudaStatus, cudaGetErrorString(cudaStatus));
			cudaFree(dev_cpart1_bitmap);
			return -1;
		}
		cudaStatus = cudaMemcpy(dev_cpart1_bitmap, sc.cluster_part1.order_bitmap, (sc.cluster_part1.cluster_bitmapnum) * sizeof(unsigned int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n", cudaStatus, cudaGetErrorString(cudaStatus));
			cudaFree(dev_cpart1_bitmap);
			cudaFree(dev_cpart1_resentry);
			return -1;
		}
		cudaStatus = cudaMemcpy(dev_cpart1_resentry, sc.cluster_part1.order_res_entry, (sc.cluster_part1.cluster_resnum) * sizeof(unsigned int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n", cudaStatus, cudaGetErrorString(cudaStatus));
			cudaFree(dev_cpart1_bitmap);
			cudaFree(dev_cpart1_resentry);
			return -1;
		}
	}
	else {
		blocksize1.x = 32;
		blocksize1.y = BLOCK_DIM_1;
		gridsize1.x = DIV(sc.cluster_part1.cluster_ordernum, BLOCK_DIM_1 * KERNEL1_PART);

		cudaStatus = cudaMalloc((void**)& dev_cpart1_orderacc, (sc.cluster_part1.cluster_ordernum + 1) * sizeof(unsigned int));
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n", cudaStatus, cudaGetErrorString(cudaStatus));
			return -1;
		}

		cudaStatus = cudaMemcpy(dev_cpart1_orderacc, sc.cluster_part1.order_accumulate_array, (sc.cluster_part1.cluster_ordernum + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n", cudaStatus, cudaGetErrorString(cudaStatus));
			cudaFree(dev_cpart1_orderacc);
			return -1;
		}

	}

	if (best.kernel2 == 1) {
		cpart2tocompress(&sc.cluster_part2);
		//sc.show();
		blocksize2.x = 32;
		blocksize2.y = BLOCK_DIM_2;
		gridsize2.x = DIV(sc.cluster_part2.cluster_blocknum, BLOCK_DIM_2 * 32);

		cudaStatus = cudaMalloc((void**)& dev_cpart2_bitmap, (sc.cluster_part2.cluster_bitmapnum) * sizeof(unsigned int));
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n", cudaStatus, cudaGetErrorString(cudaStatus));
			return -1;
		}
		cudaStatus = cudaMalloc((void**)& dev_cpart2_resentry, (sc.cluster_part2.cluster_resnum) * sizeof(unsigned int));
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n", cudaStatus, cudaGetErrorString(cudaStatus));
			cudaFree(dev_cpart2_bitmap);
			return -1;
		}
		cudaStatus = cudaMemcpy(dev_cpart2_bitmap, sc.cluster_part2.order_bitmap, (sc.cluster_part2.cluster_bitmapnum) * sizeof(unsigned int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n", cudaStatus, cudaGetErrorString(cudaStatus));
			cudaFree(dev_cpart2_bitmap);
			cudaFree(dev_cpart2_resentry);
			return -1;
		}
		cudaStatus = cudaMemcpy(dev_cpart2_resentry, sc.cluster_part2.order_res_entry, (sc.cluster_part2.cluster_resnum) * sizeof(unsigned int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n", cudaStatus, cudaGetErrorString(cudaStatus));
			cudaFree(dev_cpart2_bitmap);
			cudaFree(dev_cpart2_resentry);
			return -1;
		}
	}
	else {
		blocksize2.x = 32;
		blocksize2.y = BLOCK_DIM_2;
		gridsize2.x = DIV(sc.cluster_part2.cluster_ordernum, BLOCK_DIM_2 * KERNEL2_PART);

		cudaStatus = cudaMalloc((void**)& dev_cpart2_orderacc, (sc.cluster_part2.cluster_ordernum + 1) * sizeof(unsigned int));
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n", cudaStatus, cudaGetErrorString(cudaStatus));
			return -1;
		}

		cudaStatus = cudaMemcpy(dev_cpart2_orderacc, sc.cluster_part2.order_accumulate_array, (sc.cluster_part2.cluster_ordernum + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n", cudaStatus, cudaGetErrorString(cudaStatus));
			cudaFree(dev_cpart2_orderacc);
			return -1;
		}

	}

	if (best.kernel3 == 1) {
		sort_dispart(&sc.discrete_part);
	}

	blocksize3.x = 32;
	blocksize3.y = BLOCK_DIM_3;
	gridsize3.x = DIV(sc.discrete_part.discrete_blocknum, BLOCK_DIM_3 * 32);

	//run kernel
	checkCudaErrors(cudaSetDevice(0));

	checkCudaErrors(cudaMalloc((void**)& dev_value_res, (sc.rows) * sizeof(dataType)));
	checkCudaErrors(cudaMalloc((void**)& dev_vector, (sc.cols) * sizeof(dataType)));
	//test make the three array in one array
	//cpart1
	checkCudaErrors(cudaMalloc((void**)& dev_cpart1_order, (sc.cluster_part1.cluster_ordernum) * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)& dev_cpart1_row, (sc.cluster_part1.cluster_blocknum) * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)& dev_cpart1_data, (sc.cluster_part1.cluster_blocknum) * sizeof(dataType)));

	//test make the three array in one array
	//cpart2
	checkCudaErrors(cudaMalloc((void**)& dev_cpart2_col, (sc.cluster_part2.cluster_ordernum * 2) * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)& dev_cpart2_row, (sc.cluster_part2.cluster_blocknum) * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)& dev_cpart2_data, (sc.cluster_part2.cluster_blocknum * 2) * sizeof(dataType)));
	//discrete part
	checkCudaErrors(cudaMalloc((void**)& dev_discrete_index, (sc.discrete_part.discrete_blocknum + 1) * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)& dev_discrete_row, (sc.discrete_part.discrete_blocknum) * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)& dev_discrete_col, (sc.discrete_part.discrete_nnznum) * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)& dev_discrete_data, (sc.discrete_part.discrete_nnznum) * sizeof(dataType)));


	checkCudaErrors(cudaMemcpy(dev_vector, x_vector, (sc.cols) * sizeof(dataType), cudaMemcpyHostToDevice));
	//cpart 1
	checkCudaErrors(cudaMemcpy(dev_cpart1_order, sc.cluster_part1.order, (sc.cluster_part1.cluster_ordernum) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_cpart1_row, sc.cluster_part1.row_index_array, (sc.cluster_part1.cluster_blocknum) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_cpart1_data, sc.cluster_part1.data, (sc.cluster_part1.cluster_blocknum) * sizeof(dataType), cudaMemcpyHostToDevice));


	//cpart 2
	checkCudaErrors(cudaMemcpy(dev_cpart2_col, sc.cluster_part2.col_index_array, (sc.cluster_part2.cluster_ordernum * 2) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_cpart2_row, sc.cluster_part2.row_index_array, (sc.cluster_part2.cluster_blocknum) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_cpart2_data, sc.cluster_part2.data, (sc.cluster_part2.cluster_blocknum * 2) * sizeof(dataType), cudaMemcpyHostToDevice));
	//discrete part
	checkCudaErrors(cudaMemcpy(dev_discrete_index, sc.discrete_part.index_accumulate_array, (sc.discrete_part.discrete_blocknum + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_discrete_row, sc.discrete_part.row_index_array, (sc.discrete_part.discrete_blocknum) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_discrete_col, sc.discrete_part.col_index_array, (sc.discrete_part.discrete_nnznum) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_discrete_data, sc.discrete_part.data, (sc.discrete_part.discrete_nnznum) * sizeof(dataType), cudaMemcpyHostToDevice));

	//testkernel<<<gridsize1,blocksize1>>>();

	//kernel run
	cudaEvent_t start, stop;
	float elapsedTime = 0;
	float kerneltime1, kerneltime2, kerneltime3, kerneltime11;

	if (best.kernel1 == 0 && best.kernel2 == 0) {
		//kernel no compress
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		/* first time
		if (gridsize1.x) {
			_spmv_cpart1_dewarp << < gridsize1, blocksize1 >> > (sc.cluster_part1.cluster_ordernum, dev_cpart1_order, dev_cpart1_orderacc, dev_cpart1_row,
				dev_cpart1_data, dev_vector, dev_value_res);
		}
		if (gridsize2.x) {
			_spmv_cpart2_dewarp << <gridsize2, blocksize2 >> > (sc.cluster_part2.cluster_ordernum, dev_cpart2_orderacc, dev_cpart2_row, dev_cpart2_col,
				dev_cpart2_data, dev_vector, dev_value_res);
		}
		if (gridsize3.x) {
			_spmv_discretepart << <gridsize3, blocksize3 >> > (sc.discrete_part.discrete_blocknum, dev_discrete_index, dev_discrete_row,
				dev_discrete_col, dev_discrete_data, dev_vector, dev_value_res);
		}
		*/
		checkCudaErrors(cudaEventRecord(start, 0));

		for (int i = 1; i < RUN_TIMES; i++) {
			if (gridsize1.x) {
				_spmv_cpart1_dewarp << < gridsize1, blocksize1 >> > (sc.cluster_part1.cluster_ordernum, dev_cpart1_order, dev_cpart1_orderacc, dev_cpart1_row,
					dev_cpart1_data, dev_vector, dev_value_res);
			}
			if (gridsize2.x) {
				_spmv_cpart2_dewarp << <gridsize2, blocksize2 >> > (sc.cluster_part2.cluster_ordernum, dev_cpart2_orderacc, dev_cpart2_row, dev_cpart2_col,
					dev_cpart2_data, dev_vector, dev_value_res);
			}
			if (gridsize3.x) {
				_spmv_discretepart << <gridsize3, blocksize3 >> > (sc.discrete_part.discrete_blocknum, dev_discrete_index, dev_discrete_row,
					dev_discrete_col, dev_discrete_data, dev_vector, dev_value_res);
			}
		}
		checkCudaErrors(cudaEventRecord(stop, 0));

		checkCudaErrors(cudaEventSynchronize(stop));

		cudaStatus = cudaEventElapsedTime(&elapsedTime, start, stop);
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n",cudaStatus, cudaGetErrorString(cudaStatus));
			cudaFree(dev_value_res);
			cudaFree(dev_vector);
			cudaFree(dev_cpart1_order);
			cudaFree(dev_cpart1_row);
			cudaFree(dev_cpart1_data);
			cudaFree(dev_cpart1_orderacc);

			cudaFree(dev_cpart2_orderacc);
			cudaFree(dev_cpart2_row);
			cudaFree(dev_cpart2_col);
			cudaFree(dev_cpart2_data);

			cudaFree(dev_discrete_row);
			cudaFree(dev_discrete_col);
			cudaFree(dev_discrete_index);
			cudaFree(dev_discrete_data);
			return -1;
		}
	}
	else if (best.kernel1 == 1 && best.kernel2 == 0) {
		//kernel bit one
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		/* first time
		if (gridsize1.x) {
			_spmv_cpart1_bit << <gridsize1, blocksize1 >> > (sc.cluster_part1.cluster_blocknum, dev_cpart1_order, dev_cpart1_resentry, dev_cpart1_bitmap,
				dev_cpart1_row, dev_cpart1_data, dev_vector, dev_value_res);
		}
		if (gridsize2.x) {
			_spmv_cpart2_dewarp << <gridsize2, blocksize2 >> > (sc.cluster_part2.cluster_ordernum, dev_cpart2_orderacc, dev_cpart2_row, dev_cpart2_col,
				dev_cpart2_data, dev_vector, dev_value_res);
		}
		if (gridsize3.x) {
			_spmv_discretepart << <gridsize3, blocksize3 >> > (sc.discrete_part.discrete_blocknum, dev_discrete_index, dev_discrete_row,
				dev_discrete_col, dev_discrete_data, dev_vector, dev_value_res);
		}
		*/
		checkCudaErrors(cudaEventRecord(start, 0));
		for (int i = 1; i < RUN_TIMES; i++) {
			if (gridsize1.x) {
				_spmv_cpart1_bit << <gridsize1, blocksize1 >> > (sc.cluster_part1.cluster_blocknum, dev_cpart1_order, dev_cpart1_resentry, dev_cpart1_bitmap,
					dev_cpart1_row, dev_cpart1_data, dev_vector, dev_value_res);
			}
			if (gridsize2.x) {
				_spmv_cpart2_dewarp << <gridsize2, blocksize2 >> > (sc.cluster_part2.cluster_ordernum, dev_cpart2_orderacc, dev_cpart2_row, dev_cpart2_col,
					dev_cpart2_data, dev_vector, dev_value_res);
			}
			if (gridsize3.x) {
				_spmv_discretepart << <gridsize3, blocksize3 >> > (sc.discrete_part.discrete_blocknum, dev_discrete_index, dev_discrete_row,
					dev_discrete_col, dev_discrete_data, dev_vector, dev_value_res);
			}
		}
		checkCudaErrors(cudaEventRecord(stop, 0));

		checkCudaErrors(cudaEventSynchronize(stop));

		cudaStatus = cudaEventElapsedTime(&elapsedTime, start, stop);
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n",cudaStatus, cudaGetErrorString(cudaStatus));
			cudaFree(dev_value_res);
			cudaFree(dev_vector);
			cudaFree(dev_cpart1_order);
			cudaFree(dev_cpart1_row);
			cudaFree(dev_cpart1_data);
			cudaFree(dev_cpart1_bitmap);
			cudaFree(dev_cpart1_resentry);

			cudaFree(dev_cpart2_orderacc);
			cudaFree(dev_cpart2_row);
			cudaFree(dev_cpart2_col);
			cudaFree(dev_cpart2_data);

			cudaFree(dev_discrete_row);
			cudaFree(dev_discrete_col);
			cudaFree(dev_discrete_index);
			cudaFree(dev_discrete_data);
			return -1;
		}
	}
	else if (best.kernel1 == 0 && best.kernel2 == 1) {
		//kernel no compress
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		/* first time
		if (gridsize1.x) {
			_spmv_cpart1_dewarp << < gridsize1, blocksize1 >> > (sc.cluster_part1.cluster_ordernum, dev_cpart1_order, dev_cpart1_orderacc, dev_cpart1_row,
				dev_cpart1_data, dev_vector, dev_value_res);
		}
		if (gridsize2.x) {
			_spmv_cpart2_bit << <gridsize2, blocksize2 >> > (sc.cluster_part2.cluster_blocknum, dev_cpart2_resentry, dev_cpart2_bitmap,
				dev_cpart2_row, dev_cpart2_col, dev_cpart2_data, dev_vector, dev_value_res);
		}
		if (gridsize3.x) {
			_spmv_discretepart << <gridsize3, blocksize3 >> > (sc.discrete_part.discrete_blocknum, dev_discrete_index, dev_discrete_row,
				dev_discrete_col, dev_discrete_data, dev_vector, dev_value_res);
		}
		*/
		checkCudaErrors(cudaEventRecord(start, 0));

		for (int i = 1; i < RUN_TIMES; i++) {
			if (gridsize1.x) {
				_spmv_cpart1_dewarp << < gridsize1, blocksize1 >> > (sc.cluster_part1.cluster_ordernum, dev_cpart1_order, dev_cpart1_orderacc, dev_cpart1_row,
					dev_cpart1_data, dev_vector, dev_value_res);
			}
			if (gridsize2.x) {
				_spmv_cpart2_bit << <gridsize2, blocksize2 >> > (sc.cluster_part2.cluster_blocknum, dev_cpart2_resentry, dev_cpart2_bitmap,
					dev_cpart2_row, dev_cpart2_col, dev_cpart2_data, dev_vector, dev_value_res);
			}
			if (gridsize3.x) {
				_spmv_discretepart << <gridsize3, blocksize3 >> > (sc.discrete_part.discrete_blocknum, dev_discrete_index, dev_discrete_row,
					dev_discrete_col, dev_discrete_data, dev_vector, dev_value_res);
			}
		}
		checkCudaErrors(cudaEventRecord(stop, 0));

		checkCudaErrors(cudaEventSynchronize(stop));

		cudaStatus = cudaEventElapsedTime(&elapsedTime, start, stop);
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n",cudaStatus, cudaGetErrorString(cudaStatus));
			cudaFree(dev_value_res);
			cudaFree(dev_vector);
			cudaFree(dev_cpart1_order);
			cudaFree(dev_cpart1_row);
			cudaFree(dev_cpart1_data);
			cudaFree(dev_cpart1_orderacc);

			cudaFree(dev_cpart2_bitmap);
			cudaFree(dev_cpart2_resentry);
			cudaFree(dev_cpart2_row);
			cudaFree(dev_cpart2_col);
			cudaFree(dev_cpart2_data);

			cudaFree(dev_discrete_row);
			cudaFree(dev_discrete_col);
			cudaFree(dev_discrete_index);
			cudaFree(dev_discrete_data);
			return -1;
		}
	}
	else if (best.kernel1 == 1 && best.kernel2 == 1) {
		//kernel bit one
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		/* first time
		if (gridsize1.x) {
			_spmv_cpart1_bit << <gridsize1, blocksize1 >> > (sc.cluster_part1.cluster_blocknum, dev_cpart1_order, dev_cpart1_resentry, dev_cpart1_bitmap,
				dev_cpart1_row, dev_cpart1_data, dev_vector, dev_value_res);
		}
		if (gridsize2.x) {
			_spmv_cpart2_bit << <gridsize2, blocksize2 >> > (sc.cluster_part2.cluster_blocknum, dev_cpart2_resentry, dev_cpart2_bitmap,
				dev_cpart2_row, dev_cpart2_col, dev_cpart2_data, dev_vector, dev_value_res);
		}
		if (gridsize3.x) {
			_spmv_discretepart << <gridsize3, blocksize3 >> > (sc.discrete_part.discrete_blocknum, dev_discrete_index, dev_discrete_row,
				dev_discrete_col, dev_discrete_data, dev_vector, dev_value_res);
		}
		*/
		checkCudaErrors(cudaEventRecord(start, 0));
		for (int i = 1; i < RUN_TIMES; i++) {
			if (gridsize1.x) {
				_spmv_cpart1_bit << <gridsize1, blocksize1 >> > (sc.cluster_part1.cluster_blocknum, dev_cpart1_order, dev_cpart1_resentry, dev_cpart1_bitmap,
					dev_cpart1_row, dev_cpart1_data, dev_vector, dev_value_res);
			}
			if (gridsize2.x) {
				_spmv_cpart2_bit << <gridsize2, blocksize2 >> > (sc.cluster_part2.cluster_blocknum, dev_cpart2_resentry, dev_cpart2_bitmap,
					dev_cpart2_row, dev_cpart2_col, dev_cpart2_data, dev_vector, dev_value_res);
			}
			if (gridsize3.x) {
				_spmv_discretepart << <gridsize3, blocksize3 >> > (sc.discrete_part.discrete_blocknum, dev_discrete_index, dev_discrete_row,
					dev_discrete_col, dev_discrete_data, dev_vector, dev_value_res);
			}

		}
		checkCudaErrors(cudaEventRecord(stop, 0));

		checkCudaErrors(cudaEventSynchronize(stop));

		cudaStatus = cudaEventElapsedTime(&elapsedTime, start, stop);
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n",cudaStatus, cudaGetErrorString(cudaStatus));
			cudaFree(dev_value_res);
			cudaFree(dev_vector);
			cudaFree(dev_cpart1_order);
			cudaFree(dev_cpart1_row);
			cudaFree(dev_cpart1_data);
			cudaFree(dev_cpart1_bitmap);
			cudaFree(dev_cpart1_resentry);

			cudaFree(dev_cpart2_bitmap);
			cudaFree(dev_cpart2_resentry);
			cudaFree(dev_cpart2_row);
			cudaFree(dev_cpart2_col);
			cudaFree(dev_cpart2_data);

			cudaFree(dev_discrete_row);
			cudaFree(dev_discrete_col);
			cudaFree(dev_discrete_index);
			cudaFree(dev_discrete_data);
			return -1;
		}
	}


	elapsedTime = elapsedTime / (RUN_TIMES - 1);

	//cout << "runtime: ms " << endl << elapsedTime << endl;
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	int flags = 1;
	for (int i = 1; i > 0; i--) {
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n", cudaStatus, cudaGetErrorString(cudaStatus));
			flags = 0;
			break;
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n", cudaStatus, cudaGetErrorString(cudaStatus));
			flags = 0;
			break;
		}
		cudaStatus = cudaMemcpy(value_res, dev_value_res, (sc.rows) * sizeof(dataType), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			//fprintf(stderr, "CudaErrors() Driver API error = %04d \"%s\".\n", cudaStatus, cudaGetErrorString(cudaStatus));
			flags = 0;
			break;
		}
	}

	checkCudaErrors(cudaFree(dev_value_res));
	checkCudaErrors(cudaFree(dev_vector));

	checkCudaErrors(cudaFree(dev_cpart1_order));
	checkCudaErrors(cudaFree(dev_cpart1_row));
	checkCudaErrors(cudaFree(dev_cpart1_data));
	if (best.kernel1 == 1) {
		checkCudaErrors(cudaFree(dev_cpart1_bitmap));
		checkCudaErrors(cudaFree(dev_cpart1_resentry));
	}
	else
		checkCudaErrors(cudaFree(dev_cpart1_orderacc));

	if (best.kernel2 == 1) {
		checkCudaErrors(cudaFree(dev_cpart2_bitmap));
		checkCudaErrors(cudaFree(dev_cpart2_resentry));
	}
	else
		checkCudaErrors(cudaFree(dev_cpart2_orderacc));

	checkCudaErrors(cudaFree(dev_cpart2_row));
	checkCudaErrors(cudaFree(dev_cpart2_col));
	checkCudaErrors(cudaFree(dev_cpart2_data));

	checkCudaErrors(cudaFree(dev_discrete_row));
	checkCudaErrors(cudaFree(dev_discrete_col));
	checkCudaErrors(cudaFree(dev_discrete_index));
	checkCudaErrors(cudaFree(dev_discrete_data));

	if (flags == 1 && elapsedTime > 0)
		return elapsedTime;
	else return -1;
}


template<class dataType>
float scootest(MTX<dataType>* mtx, SCOO<dataType> sc, dataType* cpu_res, dataType* vectornum, Plan plan) {
	dataType* gpu_res = new dataType[mtx->rows];
	memset(gpu_res, 0, mtx->rows * sizeof(dataType));

	float curtime = scoo_spmv(sc, vectornum, gpu_res, plan);
	for (int i = 0; i < mtx->rows; i++) {
		gpu_res[i] /= (RUN_TIMES - 1);
	}

	int ifcorrect = check_res(cpu_res, gpu_res, mtx->rows);
	delete[] gpu_res;

	if (curtime == -1 || ifcorrect != 1)
		return -1;
	else return curtime;
}


template<class dataType>
float tuning(MTX<dataType> * mtx, MTX<dataType> * remtx, dataType * vectornum, dataType * revector, Plan * best) {
	int rearrange = 0;
	float mintime = FLT_MAX;
	Plan plan;
	

	dataType* cpu_res = new dataType[mtx->rows];
	memset(cpu_res, 0, mtx->rows * sizeof(dataType));

	cpu_spmv(mtx, vectornum, cpu_res);

	int k1_th = 1;
	int k2_th = 1;

	//block_length > 2
	for (int bl = 2; bl <= 3; bl += 1) {
		SCOO<float> sc;
		int scoomap_num = mtx2scoo(mtx, bl, &sc);
		for (int block_dim1 = 2; block_dim1 <= 8; block_dim1 *= 2) {
			for (int block_dim2 = 2; block_dim2 <= 8; block_dim2 *= 2) {
				for (int block_dim3 = 2; block_dim3 <= 8; block_dim3 *= 2) {
					for (int kernel1 = 0; kernel1 <= 1; kernel1++) {
						for (int kernel2 = 0; kernel2 <= 1; kernel2++) {
							for (int kernel3 = 0; kernel3 <= 1; kernel3++) {
								if (kernel1 == 0) {
									for (k1_th = 2; k1_th <= 32; k1_th *= 2) {
										if (kernel2 == 0) {
											for (k2_th = 2; k2_th <= 32; k2_th *= 2) {
												plan.assignment(rearrange, bl, block_dim1, block_dim2, block_dim3, kernel1, kernel2, kernel3, k1_th, k2_th);
												float curtime = scootest(mtx, sc, cpu_res, vectornum, plan);
												if (curtime == -1)
													continue;
												//cout << "run time:" << curtime << endl;

												if (curtime < mintime) {
													//if (curtime < mintime){
													mintime = curtime;
													best->assignment(rearrange, bl, block_dim1, block_dim2, block_dim3, kernel1, kernel2, kernel3, k1_th, k2_th);
													//best->show();
												}
											}
										}
										else
										{
											plan.assignment(rearrange, bl, block_dim1, block_dim2, block_dim3, kernel1, kernel2, kernel3, k1_th, k2_th);
											float curtime = scootest(mtx, sc, cpu_res, vectornum, plan);
											if (curtime == -1)
												continue;
											//cout << "run time:" << curtime << endl;

											if (curtime < mintime) {
												//if (curtime < mintime){
												mintime = curtime;
												best->assignment(rearrange, bl, block_dim1, block_dim2, block_dim3, kernel1, kernel2, kernel3, k1_th, k2_th);
												//best->show();
											}
										}
									}
								}
								else if (kernel2 == 0) {
									for (k2_th = 2; k2_th <= 32; k2_th *= 2) {
										plan.assignment(rearrange, bl, block_dim1, block_dim2, block_dim3, kernel1, kernel2, kernel3, k1_th, k2_th);
										float curtime = scootest(mtx, sc, cpu_res, vectornum, plan);
										if (curtime == -1)
											continue;
										//cout << "run time:" << curtime << endl;

										if (curtime < mintime) {
											//if (curtime < mintime){
											mintime = curtime;
											best->assignment(rearrange, bl, block_dim1, block_dim2, block_dim3, kernel1, kernel2, kernel3, k1_th, k2_th);
											//best->show();
										}
									}
								}
								else
								{
									plan.assignment(rearrange, bl, block_dim1, block_dim2, block_dim3, kernel1, kernel2, kernel3, k1_th, k2_th);
									float curtime = scootest(mtx, sc, cpu_res, vectornum, plan);
									if (curtime == -1)
										continue;
									//cout << "run time:" << curtime << endl;

									if (curtime < mintime) {
										//if (curtime < mintime){
										mintime = curtime;
										best->assignment(rearrange, bl, block_dim1, block_dim2, block_dim3, kernel1, kernel2, kernel3, k1_th, k2_th);
										//best->show();
									}
								}
							}
						}
					}
				}
			}
		}
	}
	rearrange = 1;
	//block_length > 2
	for (int bl = 2; bl <= 4; bl += 1) {
		SCOO<float> sc;
		int scoomap_num = mtx2scoo(remtx, bl, &sc);
		for (int block_dim1 = 2; block_dim1 <= 8; block_dim1 *= 2) {
			for (int block_dim2 = 2; block_dim2 <= 8; block_dim2 *= 2) {
				for (int block_dim3 = 2; block_dim3 <= 8; block_dim3 *= 2) {
					for (int kernel1 = 0; kernel1 <= 1; kernel1++) {
						for (int kernel2 = 0; kernel2 <= 1; kernel2++) {
							for (int kernel3 = 0; kernel3 <= 1; kernel3++) {
								if (kernel1 == 0) {
									for (k1_th = 2; k1_th <= 32; k1_th *= 2) {
										if (kernel2 == 0) {
											for (k2_th = 2; k2_th <= 32; k2_th *= 2) {
												plan.assignment(rearrange, bl, block_dim1, block_dim2, block_dim3, kernel1, kernel2, kernel3, k1_th, k2_th);
												float curtime = scootest(remtx, sc, cpu_res, revector, plan);
												if (curtime == -1)
													continue;
												//cout << "run time:" << curtime << endl;

												if (curtime < mintime) {
													//if (curtime < mintime){
													mintime = curtime;
													best->assignment(rearrange, bl, block_dim1, block_dim2, block_dim3, kernel1, kernel2, kernel3, k1_th, k2_th);
													//best->show();
												}
											}
										}
										else
										{
											plan.assignment(rearrange, bl, block_dim1, block_dim2, block_dim3, kernel1, kernel2, kernel3, k1_th, k2_th);
											float curtime = scootest(remtx, sc, cpu_res, revector, plan);
											if (curtime == -1)
												continue;
											//cout << "run time:" << curtime << endl;

											if (curtime < mintime) {
												//if (curtime < mintime){
												mintime = curtime;
												best->assignment(rearrange, bl, block_dim1, block_dim2, block_dim3, kernel1, kernel2, kernel3, k1_th, k2_th);
												//best->show();
											}
										}
									}
								}
								else if (kernel2 == 0) {
									for (k2_th = 2; k2_th <= 32; k2_th *= 2) {
										plan.assignment(rearrange, bl, block_dim1, block_dim2, block_dim3, kernel1, kernel2, kernel3, k1_th, k2_th);
										float curtime = scootest(remtx, sc, cpu_res, revector, plan);
										if (curtime == -1)
											continue;
										//cout << "run time:" << curtime << endl;

										if (curtime < mintime) {
											//if (curtime < mintime){
											mintime = curtime;
											best->assignment(rearrange, bl, block_dim1, block_dim2, block_dim3, kernel1, kernel2, kernel3, k1_th, k2_th);
											//best->show();
										}
									}
								}
								else
								{
									plan.assignment(rearrange, bl, block_dim1, block_dim2, block_dim3, kernel1, kernel2, kernel3, k1_th, k2_th);
									float curtime = scootest(remtx, sc, cpu_res, revector, plan);
									if (curtime == -1)
										continue;
									//cout << "run time:" << curtime << endl;

									if (curtime < mintime) {
										//if (curtime < mintime){
										mintime = curtime;
										best->assignment(rearrange, bl, block_dim1, block_dim2, block_dim3, kernel1, kernel2, kernel3, k1_th, k2_th);
										//best->show();
									}
								}
							}
						}
					}
				}
			}
		}
	}

	delete[] cpu_res;

	if (mintime < FLT_MAX)
		return mintime;
	else
		return -1;
}


template<class dataType>
float test_cuda(
    float *d_D, float *d_ggeo, float *d_lambda0, float *d_lambda1, float *d_q, float *d_Aq, float *Aq, 
    SCOO<dataType> sc, dataType * x_vector, dataType * value_res, Plan best,
    cudaStream_t stream_spmv, cudaStream_t stream_gemm
) {
    cudaError_t cudaStatus;

	assignblockdim1(best);
	assignblockdim2(best);
	assignblockdim3(best);
	assignkernel1_th(best);
	assignkernel2_th(best);

	dim3 blocksize1, blocksize2, blocksize3;
	dim3 gridsize1, gridsize2, gridsize3;

	dataType* dev_vector = 0x0;
	dataType* dev_value_res = 0x0;
	//cpart1
	unsigned int* dev_cpart1_order = 0x0;
	unsigned int* dev_cpart1_row = 0x0;
	unsigned int* dev_cpart1_bitmap = 0x0;
	unsigned int* dev_cpart1_resentry = 0x0;
	unsigned int* dev_cpart1_orderacc = 0x0;
	dataType* dev_cpart1_data = 0x0;

	//cpart2
	unsigned int* dev_cpart2_orderacc = 0x0;
	unsigned int* dev_cpart2_row = 0x0;
	unsigned int* dev_cpart2_col = 0x0;
	unsigned int* dev_cpart2_bitmap = 0x0;
	unsigned int* dev_cpart2_resentry = 0x0;
	dataType* dev_cpart2_data = 0x0;

	//discrete part
	unsigned int* dev_discrete_row = 0x0;
	unsigned int* dev_discrete_col = 0x0;
	unsigned int* dev_discrete_index = 0x0;
	dataType* dev_discrete_data = 0x0;

    if (best.kernel1 == 1) {
		cpart1tocompress(&sc.cluster_part1);

		blocksize1.x = 32;
		blocksize1.y = BLOCK_DIM_1;
		gridsize1.x = DIV(sc.cluster_part1.cluster_blocknum, BLOCK_DIM_1 * 32);

		CHECK_CUDA(cudaMallocAsync((void**)& dev_cpart1_bitmap, (sc.cluster_part1.cluster_bitmapnum) * sizeof(unsigned int), stream_spmv));
		CHECK_CUDA(cudaMallocAsync((void**)& dev_cpart1_resentry, (sc.cluster_part1.cluster_resnum) * sizeof(unsigned int), stream_spmv));
		CHECK_CUDA(cudaMemcpyAsync(dev_cpart1_bitmap, sc.cluster_part1.order_bitmap, (sc.cluster_part1.cluster_bitmapnum) * sizeof(unsigned int), cudaMemcpyHostToDevice, stream_spmv));
		CHECK_CUDA(cudaMemcpyAsync(dev_cpart1_resentry, sc.cluster_part1.order_res_entry, (sc.cluster_part1.cluster_resnum) * sizeof(unsigned int), cudaMemcpyHostToDevice, stream_spmv));
	}
	else {
		blocksize1.x = 32;
		blocksize1.y = BLOCK_DIM_1;
		gridsize1.x = DIV(sc.cluster_part1.cluster_ordernum, BLOCK_DIM_1 * KERNEL1_PART);

		CHECK_CUDA(cudaMallocAsync((void**)& dev_cpart1_orderacc, (sc.cluster_part1.cluster_ordernum + 1) * sizeof(unsigned int), stream_spmv));
		CHECK_CUDA(cudaMemcpyAsync(dev_cpart1_orderacc, sc.cluster_part1.order_accumulate_array, (sc.cluster_part1.cluster_ordernum + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice, stream_spmv));
	}

    if (best.kernel2 == 1) {
		cpart2tocompress(&sc.cluster_part2);
		//sc.show();
		blocksize2.x = 32;
		blocksize2.y = BLOCK_DIM_2;
		gridsize2.x = DIV(sc.cluster_part2.cluster_blocknum, BLOCK_DIM_2 * 32);

		CHECK_CUDA(cudaMallocAsync((void**)& dev_cpart2_bitmap, (sc.cluster_part2.cluster_bitmapnum) * sizeof(unsigned int), stream_spmv));
		CHECK_CUDA(cudaMallocAsync((void**)& dev_cpart2_resentry, (sc.cluster_part2.cluster_resnum) * sizeof(unsigned int), stream_spmv));
		CHECK_CUDA(cudaMemcpyAsync(dev_cpart2_bitmap, sc.cluster_part2.order_bitmap, (sc.cluster_part2.cluster_bitmapnum) * sizeof(unsigned int), cudaMemcpyHostToDevice, stream_spmv));
		CHECK_CUDA(cudaMemcpyAsync(dev_cpart2_resentry, sc.cluster_part2.order_res_entry, (sc.cluster_part2.cluster_resnum) * sizeof(unsigned int), cudaMemcpyHostToDevice, stream_spmv));
	}
	else {
		blocksize2.x = 32;
		blocksize2.y = BLOCK_DIM_2;
		gridsize2.x = DIV(sc.cluster_part2.cluster_ordernum, BLOCK_DIM_2 * KERNEL2_PART);

        CHECK_CUDA(cudaMallocAsync((void**)& dev_cpart2_orderacc, (sc.cluster_part2.cluster_ordernum + 1) * sizeof(unsigned int), stream_spmv));
		CHECK_CUDA(cudaMemcpyAsync(dev_cpart2_orderacc, sc.cluster_part2.order_accumulate_array, (sc.cluster_part2.cluster_ordernum + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice, stream_spmv));
	}

	if (best.kernel3 == 1) {
		sort_dispart(&sc.discrete_part);
	}

	blocksize3.x = 32;
	blocksize3.y = BLOCK_DIM_3;
	gridsize3.x = DIV(sc.discrete_part.discrete_blocknum, BLOCK_DIM_3 * 32);

	//run kernel
	CHECK_CUDA(cudaSetDevice(0));

    CHECK_CUDA(cudaMallocAsync((void**)& dev_value_res, (sc.rows) * sizeof(dataType), stream_spmv));
    CHECK_CUDA(cudaMallocAsync((void**)& dev_vector, (sc.cols) * sizeof(dataType), stream_spmv));
    //test make the three array in one array
    //cpart1
    CHECK_CUDA(cudaMallocAsync((void**)& dev_cpart1_order, (sc.cluster_part1.cluster_ordernum) * sizeof(unsigned int), stream_spmv));
    CHECK_CUDA(cudaMallocAsync((void**)& dev_cpart1_row, (sc.cluster_part1.cluster_blocknum) * sizeof(unsigned int), stream_spmv));
    CHECK_CUDA(cudaMallocAsync((void**)& dev_cpart1_data, (sc.cluster_part1.cluster_blocknum) * sizeof(dataType), stream_spmv));

    //test make the three array in one array
    //cpart2
    CHECK_CUDA(cudaMallocAsync((void**)& dev_cpart2_col, (sc.cluster_part2.cluster_ordernum * 2) * sizeof(unsigned int), stream_spmv));
    CHECK_CUDA(cudaMallocAsync((void**)& dev_cpart2_row, (sc.cluster_part2.cluster_blocknum) * sizeof(unsigned int), stream_spmv));
    CHECK_CUDA(cudaMallocAsync((void**)& dev_cpart2_data, (sc.cluster_part2.cluster_blocknum * 2) * sizeof(dataType), stream_spmv));
    //discrete part
    CHECK_CUDA(cudaMallocAsync((void**)& dev_discrete_index, (sc.discrete_part.discrete_blocknum + 1) * sizeof(unsigned int), stream_spmv));
    CHECK_CUDA(cudaMallocAsync((void**)& dev_discrete_row, (sc.discrete_part.discrete_blocknum) * sizeof(unsigned int), stream_spmv));
    CHECK_CUDA(cudaMallocAsync((void**)& dev_discrete_col, (sc.discrete_part.discrete_nnznum) * sizeof(unsigned int), stream_spmv));
    CHECK_CUDA(cudaMallocAsync((void**)& dev_discrete_data, (sc.discrete_part.discrete_nnznum) * sizeof(dataType), stream_spmv));

    CHECK_CUDA(cudaMemcpyAsync(dev_vector, x_vector, (sc.cols) * sizeof(dataType), cudaMemcpyHostToDevice, stream_spmv));
    //cpart 1
    CHECK_CUDA(cudaMemcpyAsync(dev_cpart1_order, sc.cluster_part1.order, (sc.cluster_part1.cluster_ordernum) * sizeof(unsigned int), cudaMemcpyHostToDevice, stream_spmv));
    CHECK_CUDA(cudaMemcpyAsync(dev_cpart1_row, sc.cluster_part1.row_index_array, (sc.cluster_part1.cluster_blocknum) * sizeof(unsigned int), cudaMemcpyHostToDevice, stream_spmv));
    CHECK_CUDA(cudaMemcpyAsync(dev_cpart1_data, sc.cluster_part1.data, (sc.cluster_part1.cluster_blocknum) * sizeof(dataType), cudaMemcpyHostToDevice, stream_spmv));

    //cpart 2
    CHECK_CUDA(cudaMemcpyAsync(dev_cpart2_col, sc.cluster_part2.col_index_array, (sc.cluster_part2.cluster_ordernum * 2) * sizeof(unsigned int), cudaMemcpyHostToDevice, stream_spmv));
    CHECK_CUDA(cudaMemcpyAsync(dev_cpart2_row, sc.cluster_part2.row_index_array, (sc.cluster_part2.cluster_blocknum) * sizeof(unsigned int), cudaMemcpyHostToDevice, stream_spmv));
    CHECK_CUDA(cudaMemcpyAsync(dev_cpart2_data, sc.cluster_part2.data, (sc.cluster_part2.cluster_blocknum * 2) * sizeof(dataType), cudaMemcpyHostToDevice, stream_spmv));
    //discrete part
    CHECK_CUDA(cudaMemcpyAsync(dev_discrete_index, sc.discrete_part.index_accumulate_array, (sc.discrete_part.discrete_blocknum + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice, stream_spmv));
    CHECK_CUDA(cudaMemcpyAsync(dev_discrete_row, sc.discrete_part.row_index_array, (sc.discrete_part.discrete_blocknum) * sizeof(unsigned int), cudaMemcpyHostToDevice, stream_spmv));
    CHECK_CUDA(cudaMemcpyAsync(dev_discrete_col, sc.discrete_part.col_index_array, (sc.discrete_part.discrete_nnznum) * sizeof(unsigned int), cudaMemcpyHostToDevice, stream_spmv));
    CHECK_CUDA(cudaMemcpyAsync(dev_discrete_data, sc.discrete_part.data, (sc.discrete_part.discrete_nnznum) * sizeof(dataType), cudaMemcpyHostToDevice, stream_spmv));

	//kernel run
	cudaEvent_t start, stop;
	float elapsedTime = 0;
	float kerneltime1, kerneltime2, kerneltime3, kerneltime11;

    //gemm
    dim3 threadsPerBlock(8, 8);
    int num_blocks = num_elements;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));
    occa_ellipticPartialAxCoeffHex3D_v1_0<<<num_blocks, threadsPerBlock, 0, stream_gemm>>>(d_D, d_ggeo, d_lambda0, d_lambda1, d_q, d_Aq);
    CHECK_CUDA(cudaMemcpyAsync(Aq, d_Aq, num_elements * 512 * sizeof(float), cudaMemcpyDeviceToHost, stream_gemm));

    //spmv
	if (best.kernel1 == 0 && best.kernel2 == 0) {
		//kernel no compress
		// CHECK_CUDA(cudaEventCreate(&start));
		// CHECK_CUDA(cudaEventCreate(&stop));
		// CHECK_CUDA(cudaEventRecord(start, 0));

		for (int i = 1; i < RUN_TIMES; i++) {
			if (gridsize1.x) {
				_spmv_cpart1_dewarp << < gridsize1, blocksize1, 0, stream_spmv >> > (sc.cluster_part1.cluster_ordernum, dev_cpart1_order, dev_cpart1_orderacc, dev_cpart1_row,
					dev_cpart1_data, dev_vector, dev_value_res);
			}
			if (gridsize2.x) {
				_spmv_cpart2_dewarp << <gridsize2, blocksize2, 0, stream_spmv >> > (sc.cluster_part2.cluster_ordernum, dev_cpart2_orderacc, dev_cpart2_row, dev_cpart2_col,
					dev_cpart2_data, dev_vector, dev_value_res);
			}
			if (gridsize3.x) {
				_spmv_discretepart << <gridsize3, blocksize3, 0, stream_spmv >> > (sc.discrete_part.discrete_blocknum, dev_discrete_index, dev_discrete_row,
					dev_discrete_col, dev_discrete_data, dev_vector, dev_value_res);
			}
		}
		CHECK_CUDA(cudaEventRecord(stop, 0));

		CHECK_CUDA(cudaEventSynchronize(stop));

		CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
	}
	else if (best.kernel1 == 1 && best.kernel2 == 0) {
		//kernel bit one
		// CHECK_CUDA(cudaEventCreate(&start));
		// CHECK_CUDA(cudaEventCreate(&stop));
		// CHECK_CUDA(cudaEventRecord(start, 0));
		for (int i = 1; i < RUN_TIMES; i++) {
			if (gridsize1.x) {
				_spmv_cpart1_bit << <gridsize1, blocksize1, 0, stream_spmv >> > (sc.cluster_part1.cluster_blocknum, dev_cpart1_order, dev_cpart1_resentry, dev_cpart1_bitmap,
					dev_cpart1_row, dev_cpart1_data, dev_vector, dev_value_res);
			}
			if (gridsize2.x) {
				_spmv_cpart2_dewarp << <gridsize2, blocksize2, 0, stream_spmv >> > (sc.cluster_part2.cluster_ordernum, dev_cpart2_orderacc, dev_cpart2_row, dev_cpart2_col,
					dev_cpart2_data, dev_vector, dev_value_res);
			}
			if (gridsize3.x) {
				_spmv_discretepart << <gridsize3, blocksize3, 0, stream_spmv >> > (sc.discrete_part.discrete_blocknum, dev_discrete_index, dev_discrete_row,
					dev_discrete_col, dev_discrete_data, dev_vector, dev_value_res);
			}
		}
		CHECK_CUDA(cudaEventRecord(stop, 0));

		CHECK_CUDA(cudaEventSynchronize(stop));

		CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
	}
	else if (best.kernel1 == 0 && best.kernel2 == 1) {
		//kernel no compress
		// CHECK_CUDA(cudaEventCreate(&start));
		// CHECK_CUDA(cudaEventCreate(&stop));
		// CHECK_CUDA(cudaEventRecord(start, 0));

		for (int i = 1; i < RUN_TIMES; i++) {
			if (gridsize1.x) {
				_spmv_cpart1_dewarp << < gridsize1, blocksize1, 0, stream_spmv >> > (sc.cluster_part1.cluster_ordernum, dev_cpart1_order, dev_cpart1_orderacc, dev_cpart1_row,
					dev_cpart1_data, dev_vector, dev_value_res);
			}
			if (gridsize2.x) {
				_spmv_cpart2_bit << <gridsize2, blocksize2, 0, stream_spmv >> > (sc.cluster_part2.cluster_blocknum, dev_cpart2_resentry, dev_cpart2_bitmap,
					dev_cpart2_row, dev_cpart2_col, dev_cpart2_data, dev_vector, dev_value_res);
			}
			if (gridsize3.x) {
				_spmv_discretepart << <gridsize3, blocksize3, 0, stream_spmv >> > (sc.discrete_part.discrete_blocknum, dev_discrete_index, dev_discrete_row,
					dev_discrete_col, dev_discrete_data, dev_vector, dev_value_res);
			}
		}
		CHECK_CUDA(cudaEventRecord(stop, 0));

		CHECK_CUDA(cudaEventSynchronize(stop));

		CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
	}
	else if (best.kernel1 == 1 && best.kernel2 == 1) {
		//kernel bit one
		// CHECK_CUDA(cudaEventCreate(&start));
		// CHECK_CUDA(cudaEventCreate(&stop));
		// CHECK_CUDA(cudaEventRecord(start, 0));
		for (int i = 1; i < RUN_TIMES; i++) {
			if (gridsize1.x) {
				_spmv_cpart1_bit << <gridsize1, blocksize1, 0, stream_spmv >> > (sc.cluster_part1.cluster_blocknum, dev_cpart1_order, dev_cpart1_resentry, dev_cpart1_bitmap,
					dev_cpart1_row, dev_cpart1_data, dev_vector, dev_value_res);
			}
			if (gridsize2.x) {
				_spmv_cpart2_bit << <gridsize2, blocksize2, 0, stream_spmv >> > (sc.cluster_part2.cluster_blocknum, dev_cpart2_resentry, dev_cpart2_bitmap,
					dev_cpart2_row, dev_cpart2_col, dev_cpart2_data, dev_vector, dev_value_res);
			}
			if (gridsize3.x) {
				_spmv_discretepart << <gridsize3, blocksize3, 0, stream_spmv >> > (sc.discrete_part.discrete_blocknum, dev_discrete_index, dev_discrete_row,
					dev_discrete_col, dev_discrete_data, dev_vector, dev_value_res);
			}

		}
		CHECK_CUDA(cudaEventRecord(stop, 0));

		CHECK_CUDA(cudaEventSynchronize(stop));

		CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
	}

	elapsedTime = elapsedTime / (RUN_TIMES - 1);

	//cout << "runtime: ms " << endl << elapsedTime << endl;
	CHECK_CUDA(cudaEventDestroy(start));
	CHECK_CUDA(cudaEventDestroy(stop));

	int flags = 1;
	for (int i = 1; i > 0; i--) {
		CHECK_CUDA(cudaGetLastError());
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaMemcpyAsync(value_res, dev_value_res, (sc.rows) * sizeof(dataType), cudaMemcpyDeviceToHost, stream_spmv));
	}
    cudaStreamSynchronize(stream_spmv);
    cudaStreamSynchronize(stream_gemm);

	CHECK_CUDA(cudaFree(dev_value_res));
	CHECK_CUDA(cudaFree(dev_vector));

	CHECK_CUDA(cudaFree(dev_cpart1_order));
	CHECK_CUDA(cudaFree(dev_cpart1_row));
	CHECK_CUDA(cudaFree(dev_cpart1_data));
	if (best.kernel1 == 1) {
		CHECK_CUDA(cudaFree(dev_cpart1_bitmap));
		CHECK_CUDA(cudaFree(dev_cpart1_resentry));
	}
	else
		CHECK_CUDA(cudaFree(dev_cpart1_orderacc));

	if (best.kernel2 == 1) {
		CHECK_CUDA(cudaFree(dev_cpart2_bitmap));
		CHECK_CUDA(cudaFree(dev_cpart2_resentry));
	}
	else
		CHECK_CUDA(cudaFree(dev_cpart2_orderacc));

	CHECK_CUDA(cudaFree(dev_cpart2_row));
	CHECK_CUDA(cudaFree(dev_cpart2_col));
	CHECK_CUDA(cudaFree(dev_cpart2_data));

	CHECK_CUDA(cudaFree(dev_discrete_row));
	CHECK_CUDA(cudaFree(dev_discrete_col));
	CHECK_CUDA(cudaFree(dev_discrete_index));
	CHECK_CUDA(cudaFree(dev_discrete_data));

	if (flags == 1 && elapsedTime > 0)
		return elapsedTime;
	else return -1;
}

int main(int argc, char* argv[])
{
    cudaStream_t stream_spmv, stream_gemm;
    cudaStreamCreate(&stream_spmv);
    cudaStreamCreate(&stream_gemm);
    // std::cout << "preparing data for ax computing..." << std::endl;

    // Allocate memory on host
    float *D, *ggeo, *lambda0, *lambda1, *q, *Aq;
    CHECK_CUDA(cudaMallocHost(&D, 64 * sizeof(float)));                        // D保持8x8矩阵
    CHECK_CUDA(cudaMallocHost(&ggeo, num_elements * 7 * 512 * sizeof(float))); // 几何参数扩展为num_elements*7*512
    CHECK_CUDA(cudaMallocHost(&lambda0, num_elements * 512 * sizeof(float)));  // lambda0扩展为num_elements*512
    CHECK_CUDA(cudaMallocHost(&lambda1, num_elements * 512 * sizeof(float)));  // lambda1扩展为num_elements*512
    CHECK_CUDA(cudaMallocHost(&q, num_elements * 512 * sizeof(float)));        // q扩展为num_elements*512
    CHECK_CUDA(cudaMallocHost(&Aq, num_elements * 512 * sizeof(float)));        // Aq扩展为num_elements*512

    // Initialize arrays on host
    initialize_arrays(D, ggeo, lambda0, lambda1, q, Aq);

    // std::cout << "allocating data for ax computing..." << std::endl;

    // Allocate memory on device
    float *d_D, *d_ggeo, *d_lambda0, *d_lambda1, *d_q, *d_Aq;
    CHECK_CUDA(cudaMallocAsync(&d_D, 64 * sizeof(float), stream_gemm));
    CHECK_CUDA(cudaMallocAsync(&d_ggeo, num_elements * 512 * 7 * sizeof(float), stream_gemm));
    CHECK_CUDA(cudaMallocAsync(&d_lambda0, num_elements * 512 * sizeof(float), stream_gemm));
    CHECK_CUDA(cudaMallocAsync(&d_lambda1, num_elements * 512 * sizeof(float), stream_gemm));
    CHECK_CUDA(cudaMallocAsync(&d_q, num_elements * 512 * sizeof(float), stream_gemm));
    CHECK_CUDA(cudaMallocAsync(&d_Aq, num_elements * 512 * sizeof(float), stream_gemm));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpyAsync(d_D, D, 64 * sizeof(float), cudaMemcpyHostToDevice, stream_gemm));
    CHECK_CUDA(cudaMemcpyAsync(d_ggeo, ggeo, num_elements * 512 * 7 * sizeof(float), cudaMemcpyHostToDevice, stream_gemm));
    CHECK_CUDA(cudaMemcpyAsync(d_lambda0, lambda0, num_elements * 512 * sizeof(float), cudaMemcpyHostToDevice, stream_gemm));
    CHECK_CUDA(cudaMemcpyAsync(d_lambda1, lambda1, num_elements * 512 * sizeof(float), cudaMemcpyHostToDevice, stream_gemm));
    CHECK_CUDA(cudaMemcpyAsync(d_q, q, num_elements * 512 * sizeof(float), cudaMemcpyHostToDevice, stream_gemm));
    CHECK_CUDA(cudaMemcpyAsync(d_Aq, Aq, num_elements * 512 * sizeof(float), cudaMemcpyHostToDevice, stream_gemm));

    // std::cout << "prepare for spmv computing..." << std::endl;

	char* filename = (char*)argv[1];
	//char filename[100] = "pdb1HYS.mtx";
	//string filename = "pdb1HYS.mtx";
	FILE* infile = fopen(filename, "r");
	cout << filename << endl;

	//fileToMtx<float>(filename, &mtx);
	//printMtx<float>(&mtx);

	MTX<float> mtx;

	int file_test = 1;
	if (file_test == 1) {
		fileToMtx<float>(filename, &mtx);
	}
	else {
		//定义
		mtx.rows = 300;
		mtx.cols = 600;
		mtx.nnz = 600;
		mtx.row = new int[mtx.nnz];
		mtx.col = new int[mtx.nnz];
		mtx.data = new float[mtx.nnz];

		for (int i = 0; i < 300; i++) {
			mtx.row[i * 2] = i;
			mtx.col[i * 2] = i;
			mtx.row[i * 2 + 1] = i;
			mtx.col[i * 2 + 1] = i + 1;
		}

		for (int i = 0; i < mtx.nnz; i++) {
			mtx.data[i] = (mtx.col[i] + mtx.row[i]) / 100.0;
		}

	}

	float* x_vector = new float[mtx.cols];
	float* revector = new float[mtx.cols];
	float* result = new float[mtx.rows];

	for (int i = 0; i < mtx.cols; i++) {
		//x_vector[i] = 1;
		x_vector[i] = (i + 1) % 100 * 100 / (float)mtx.cols;
	}
	MTX<float> remtx;
	rearrangecol(&mtx, &remtx, x_vector, revector);
	// Plan to Automatic tuning
	Plan best;
	//Automatic tuning or no tuning, when tune=1 tuning
	int tune = 0;
	float tuning_result = -1;
	if (tune == 0) {
		char config_filename[100] = "optimal.cfg";
		FILE* infile_1 = fopen(config_filename, "r");
		fscanf(infile_1, "%d", &best.rearrange);
		fscanf(infile_1, "%d", &best.block_length);
		fscanf(infile_1, "%d", &best.block_dim_1);
		fscanf(infile_1, "%d", &best.block_dim_2);
		fscanf(infile_1, "%d", &best.block_dim_3);
		fscanf(infile_1, "%d", &best.kernel1);
		fscanf(infile_1, "%d", &best.kernel2);
		fscanf(infile_1, "%d", &best.kernel3);
		fscanf(infile_1, "%d", &best.kernel1_thread);
		fscanf(infile_1, "%d", &best.kernel2_thread);
		fclose(infile_1);
		tuning_result = 1;
	}
	else
		tuning_result = tuning(&mtx, &remtx, x_vector, revector, &best);
	if (tuning_result > 0) {
		// best.show();
		SCOO<float> sc;
		int scoomap_num;
		struct timeval ts, te;
		if(best.rearrange == 0){
			gettimeofday(&ts, NULL);
		 	scoomap_num = mtx2scoo(&mtx, best.block_length, &sc);
			gettimeofday(&te, NULL);	
		}
		else{
			gettimeofday(&ts, NULL);
			scoomap_num = mtx2scoo(&remtx, best.block_length, &sc);
			gettimeofday(&te, NULL);	
		}
		// cout << "reordering time: " << 1e3 * (te.tv_sec - ts.tv_sec) + (te.tv_usec - ts.tv_usec) / 1e3 << endl;
		// sc.show();
		
		// cout << "The best run time when tuning:(ms)" << endl << tuning_result << endl;
		float run_time;
		// if(best.rearrange == 0){
		//  	run_time = scoo_spmv(sc, x_vector, result, best);
		// }
		// else{
		// 	run_time = scoo_spmv(sc, revector, result, best);
		// }
		
		// cout << "The best run time: (ms) " << endl << run_time << endl;

        // run fused kernel
        if(best.rearrange == 0){
            run_time = test_cuda(d_D, d_ggeo, d_lambda0, d_lambda1, d_q, d_Aq, Aq, sc, x_vector, result, best, stream_spmv, stream_gemm);
		}
		else{
			run_time = test_cuda(d_D, d_ggeo, d_lambda0, d_lambda1, d_q, d_Aq, Aq, sc, revector, result, best, stream_spmv, stream_gemm);
		}
        // cout << "The fused run time: (ms) " << endl;
        cout << run_time << endl;


		for (int i = 0; i < mtx.rows; i++) {
			result[i] /= (RUN_TIMES - 1);
		}

		float* cres = new float[mtx.rows];
		memset(cres, 0, mtx.rows * sizeof(float));
		cpu_spmv(&mtx, x_vector, cres);
		int iscorrect = check_res(cres, result, mtx.rows);
		if (iscorrect == 1)
			cout << "Correct!" << endl;
		else
			cout << "Error!" << endl;
		if(best.kernel1 == 1){
			cpart1tocompress(&sc.cluster_part1);
		}
		if(best.kernel2 == 1){
			cpart2tocompress(&sc.cluster_part2);
		}
		calstorage(sc, best);
		cusparse_spmv(mtx, x_vector);

		delete[] cres;
	}
	else
	{
		cout << "Error when tuning!" << endl;
	}
        // Free memory
    CHECK_CUDA(cudaFree(d_D));
    CHECK_CUDA(cudaFree(d_ggeo));
    CHECK_CUDA(cudaFree(d_lambda0));
    CHECK_CUDA(cudaFree(d_lambda1));
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_Aq));
    CHECK_CUDA(cudaFreeHost(D));
    CHECK_CUDA(cudaFreeHost(ggeo));
    CHECK_CUDA(cudaFreeHost(lambda0));
    CHECK_CUDA(cudaFreeHost(lambda1));
    CHECK_CUDA(cudaFreeHost(q));
    CHECK_CUDA(cudaFreeHost(Aq));

	/*
	for (int i = 0; i < mtx.rows; i++) {
		cout << "num:" << i << ",results = " << result[i] << endl;
	}
	*/
	delete[] x_vector;
	delete[] revector;
	delete[] result;

	return 0;
}