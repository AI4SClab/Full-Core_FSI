#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream> 
#include <stdlib.h>
#include <cusparse.h>
#include<cstring>
#include<iostream>
#include "mtx.hpp"
using namespace std;

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#ifndef checkCudaErrors
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, const char* file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
			err, cudaGetErrorString(err), file, line);//getCudaDrvErrorString
		exit(EXIT_FAILURE);
	}
}
#endif

#ifndef checkCusparseErrors
#define checkCusparseErrors(err)  __checkCusparseErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCusparseErrors(cusparseStatus_t err, const char* file, const int line)
{
	if (CUSPARSE_STATUS_SUCCESS != err)
	{
		fprintf(stderr, "checkCusparseErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
			err, cusparseGetErrorString(err), file, line);//getCudaDrvErrorString
		exit(EXIT_FAILURE);
	}
}
#endif


template<class dataType>
void cusparse_spmv(MTX<dataType> mtx, dataType * vector) {
	const int A_num_rows = mtx.rows;
	const int A_num_cols = mtx.cols;
	const int A_nnz = mtx.nnz;

	dataType* hA_values = new dataType[mtx.nnz];
	int* hA_columns = new int[mtx.nnz];
	int* hA_csrOffsets = new int[mtx.rows + 1];

	coo2csr(&mtx, hA_values, hA_columns, hA_csrOffsets);

	// Host problem definition
	dataType* hY = new dataType[mtx.rows];
	for (int i = 0; i < mtx.rows; i++) {
		hY[i] = 0.0f;
	}

	float     alpha = 1.0f;
	float     beta = 0.0f;

	// Device memory management
	int* dA_csrOffsets, * dA_columns;
	dataType* dA_values, * dX, * dY;

	checkCudaErrors(cudaMalloc((void**)& dA_csrOffsets, (A_num_rows + 1) * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)& dA_columns, A_nnz * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)& dA_values, A_nnz * sizeof(dataType)));
	checkCudaErrors(cudaMalloc((void**)& dX, A_num_cols * sizeof(dataType)));
	checkCudaErrors(cudaMalloc((void**)& dY, A_num_rows * sizeof(dataType)));

	checkCudaErrors(cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
		(A_num_rows + 1) * sizeof(int),
		cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
		cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(dataType),
		cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dX, vector, A_num_cols * sizeof(dataType),
		cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dY, hY, A_num_rows * sizeof(dataType),
		cudaMemcpyHostToDevice));

	//--------------------------------------------------------------------------
	// CUSPARSE APIs
	cusparseHandle_t     handle = NULL;
	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecX, vecY;
	void* dBuffer = NULL;
	size_t               bufferSize = 0;
	checkCusparseErrors(cusparseCreate(&handle));
	checkCusparseErrors(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
		dA_csrOffsets, dA_columns, dA_values,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

	checkCusparseErrors(cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F));
	checkCusparseErrors(cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F));

	checkCusparseErrors(cusparseSpMV_bufferSize(
		handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
		CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
	checkCudaErrors(cudaMalloc(&dBuffer, bufferSize));

	//kernel run
	cudaEvent_t start, stop;
	float elapsedTime;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	/* first time 
	checkCusparseErrors(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
		CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
	*/
	checkCudaErrors(cudaEventRecord(start, 0));

	for (int i = 1; i < RUN_TIMES; i++) {
		checkCusparseErrors(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
			CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
	}

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));

	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	elapsedTime = elapsedTime / (RUN_TIMES - 1);

	cout << "cusparse runtime: ms " << endl << elapsedTime << endl;

	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	checkCusparseErrors(cusparseDestroySpMat(matA));
	checkCusparseErrors(cusparseDestroyDnVec(vecX));
	checkCusparseErrors(cusparseDestroyDnVec(vecY));
	checkCusparseErrors(cusparseDestroy(handle));

	checkCudaErrors(cudaMemcpy(hY, dY, A_num_rows * sizeof(dataType),
		cudaMemcpyDeviceToHost));

	dataType* hY_result = new dataType[mtx.rows];
	cout << "cusparse test:\n";

	cpu_spmv(&mtx, vector, hY_result);
    
	int iscorrect = check_res(hY_result, hY, mtx.rows);
    if(iscorrect == 1)
        cout<<"Correct!"<<endl;
    else
        cout<<"Error!"<<endl;

	//--------------------------------------------------------------------------
	// device memory deallocation
	delete mtx.data;
	delete hA_csrOffsets;
	delete hA_columns;
	delete hA_values;
	delete hY;

	checkCudaErrors(cudaFree(dBuffer));
	checkCudaErrors(cudaFree(dA_csrOffsets));
	checkCudaErrors(cudaFree(dA_columns));
	checkCudaErrors(cudaFree(dA_values));
	checkCudaErrors(cudaFree(dX));
	checkCudaErrors(cudaFree(dY));
}
