#pragma once
#include <stdio.h>
#include <stdlib.h>
using namespace std;

#include "mtx.hpp"

template<class dataType>
void cpu_spmv(MTX<dataType>* mtx, dataType* vec, dataType* res)
{
	int row = mtx->row[0], row2;
	dataType sum = 0;
	for (int i = 0; i < mtx->nnz; i++) {
		row2 = mtx->row[i];
		if (row2 != row) {
			res[row] = sum;
			sum = 0;
		}
		row = row2;
		sum += mtx->data[i] * vec[mtx->col[i]];
	}
	res[row2] = sum;
}

template<class dataType>
int check_res(dataType* cres, dataType* res, int len)
{
	int count = 0;
	dataType max = 0.0, ratio = 0.0;
	// printf("len=%d\n",len);
	for (int i = 0; i < len; i++) {
		dataType minus = cres[i] - res[i];
		if (minus < 0) minus = 0 - minus;
		dataType v = cres[i] > 0 ? cres[i] : 0 - cres[i];

		//if(count == 0 && i<10)
		//if (i < 100)
			//cout << "Show count: " << i << ": cpu " << cres[i] << " gpu " << res[i] << " line " << i << endl;

		// if(minus>v/1000){// minus/v > 0.1
		if (minus > 0.1) {
			count++;
			if (count < 10)
				//cout << "  Error count " << count << ": cpu " << cres[i] << " gpu " << res[i] << " line " << i << endl;
				if (v == 0) v++;
			ratio = ratio < minus / v ? minus / v : ratio;
			max = max < minus ? minus : max;
		}
		// else
		//     cout<<endl;
	}
	if (ratio > 0.01) {
		//cout << "Max diff:" << max << "  ratio:" << ratio << "  count:" << count << endl;
		return 0;
	}
	else {
		//cout << "Correct!" << endl;
		return 1;
	}
	//cout << "Correct!  ratio:" << ratio << "  count:" << count << endl;
}




template<class dataType>
bool coo2csr(MTX<dataType> * mtx, float* csr_val, int* csr_col, int* csr_row)
{
	for (int i = 0; i < mtx->rows + 1; i++) {
		csr_row[i] = 0;
	}

	for (int i = 0; i < mtx->nnz; i++)
	{
		csr_val[i] = mtx->data[i];

		csr_col[i] = mtx->col[i];
		csr_row[mtx->row[i] + 1]++;
		// if (i==0) printf("cccc, %d,%d, \n",mtx->row[i] + 1, csr_row[mtx->row[i] + 1]);
	}
	for (int i = 0; i < mtx->rows; i++)
	{
		csr_row[i + 1] += csr_row[i];
	}
	// cout << "coo2csr Done." << endl;
	return true;
}
