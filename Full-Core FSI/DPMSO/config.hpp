#pragma once

#define RUN_TIMES 11

#include <stdio.h>
#include <iostream> 
#include <stdlib.h>
using namespace std;

typedef struct Plan {
	int block_length;
	int rearrange;

	int block_dim_1;
	int block_dim_2;
	int block_dim_3;

	int kernel1;
	int kernel2;
	int kernel3;

	int kernel1_thread;
	int kernel2_thread;

	void assignment(int rea, int bl, int block_dim1, int block_dim2, int block_dim3, int k1, int k2, int k3, int k1_th, int k2_th) {
		rearrange = rea;
		block_length = bl;

		block_dim_1 = block_dim1;
		block_dim_2 = block_dim2;
		block_dim_3 = block_dim3;

		kernel1 = k1;
		kernel2 = k2;
		kernel3 = k3;

		kernel1_thread = k1_th;
		kernel2_thread = k2_th;
	}

	void show() {
		cout << "Current Plan:" << endl;
		cout << "Rearrange:\n" << rearrange << endl;
		cout << "The matrix block length is: \n" << block_length << endl;
		switch (kernel1)
		{
		case 0:
			cout << "kernel1 is \n" << kernel1 << endl;
			cout << ",the warp num in one block is \n" << block_dim_1 << endl;
			cout << kernel1_thread << endl << "threads calculate one clusters." << endl;
			break;
		case 1:
			cout << "kernel1 is \n" << kernel1 << endl;
			cout << ",the warp num in one block is \n" << block_dim_1 << endl;
			cout << kernel1_thread << endl << "One element block calculated in one thread. The order information is compressed in bit map." << endl;
			break;
		default:
			break;
		}

		switch (kernel2)
		{
		case 0:
			cout << "kernel2 is \n" << kernel2 << endl;
			cout << ",the warp num in one block is \n" << block_dim_2 << endl;
			cout << kernel2_thread << endl <<"threads calculate one clusters." << endl;
			break;
		case 1:
			cout << "kernel2 is \n" << kernel2 << endl;
			cout << ",the warp num in one block is \n" << block_dim_2 << endl;
			cout << kernel2_thread << endl <<"Two elements block calculated in one thread. The order information is compressed in bit map." << endl;
			break;
		default:
			break;
		}

		switch (kernel3)
		{
		case 0:
			cout << "kernel3 is \n" << kernel3 << endl;
			cout << ",the warp num in one block is \n" << block_dim_3 << endl;
			cout << "One discrete block calculated in one thread." << endl;
			break;
		case 1:
			cout << "kernel3 is \n" << kernel3 << endl;
			cout << ",the warp num in one block is \n" << block_dim_3 << endl;
			cout << "The discrete blocks is sorted and calculated in one thread." << endl;
			break;
		default:
			break;
		}
	}
};

void assignblockdim1(Plan best) {
	if (best.block_dim_1 == 2) {
#ifndef BLOCK_DIM_1
#define BLOCK_DIM_1 2
#else
#undef BLOCK_DIM_1
#define BLOCK_DIM_1 2
#endif // !BLOCK_DIM
	}
	else if (best.block_dim_1 == 4) {
#ifndef BLOCK_DIM_1
#define BLOCK_DIM_1 4
#else
#undef BLOCK_DIM_1
#define BLOCK_DIM_1 4
#endif // !BLOCK_DIM
	}
	else if (best.block_dim_1 == 8) {
#ifndef BLOCK_DIM_1
#define BLOCK_DIM_1 8
#else
#undef BLOCK_DIM_1
#define BLOCK_DIM_1 8
#endif // !BLOCK_DIM
	}
	else if (best.block_dim_1 == 16) {
#ifndef BLOCK_DIM_1
#define BLOCK_DIM_1 16
#else
#undef BLOCK_DIM_1
#define BLOCK_DIM_1 16
#endif // !BLOCK_DIM
	}
}

void assignblockdim2(Plan best) {
	if (best.block_dim_2 == 2) {
#ifndef BLOCK_DIM_2
#define BLOCK_DIM_2 2
#else
#undef BLOCK_DIM_2
#define BLOCK_DIM_2 2
#endif // !BLOCK_DIM
	}
	else if (best.block_dim_2 == 4) {
#ifndef BLOCK_DIM_2
#define BLOCK_DIM_2 4
#else
#undef BLOCK_DIM_2
#define BLOCK_DIM_2 4
#endif // !BLOCK_DIM
	}
	else if (best.block_dim_2 == 8) {
#ifndef BLOCK_DIM_2
#define BLOCK_DIM_2 8
#else
#undef BLOCK_DIM_2
#define BLOCK_DIM_2 8
#endif // !BLOCK_DIM
	}
	else if (best.block_dim_2 == 16) {
#ifndef BLOCK_DIM_2
#define BLOCK_DIM_2 16
#else
#undef BLOCK_DIM_2
#define BLOCK_DIM_2 16
#endif // !BLOCK_DIM
	}
}


void assignblockdim3(Plan best) {
	if (best.block_dim_3 == 2) {
#ifndef BLOCK_DIM_3
#define BLOCK_DIM_3 2
#else
#undef BLOCK_DIM_3
#define BLOCK_DIM_3 2
#endif // !BLOCK_DIM
	}
	else if (best.block_dim_3 == 4) {
#ifndef BLOCK_DIM_3
#define BLOCK_DIM_3 4
#else
#undef BLOCK_DIM_3
#define BLOCK_DIM_3 4
#endif // !BLOCK_DIM
	}
	else if (best.block_dim_3 == 8) {
#ifndef BLOCK_DIM_3
#define BLOCK_DIM_3 8
#else
#undef BLOCK_DIM_3
#define BLOCK_DIM_3 8
#endif // !BLOCK_DIM
	}
	else if (best.block_dim_3 == 16) {
#ifndef BLOCK_DIM_3
#define BLOCK_DIM_3 16
#else
#undef BLOCK_DIM_3
#define BLOCK_DIM_3 16
#endif // !BLOCK_DIM
	}
}

void assignkernel1_th(Plan best) {
	if (best.kernel1_thread == 2) {
#ifndef KERNEL1_PART
#define KERNEL1_PART 16
#else
#undef KERNEL1_PART
#define KERNEL1_PART 16
#endif // !BLOCK_DIM
	}
	else if (best.kernel1_thread == 4) {
#ifndef KERNEL1_PART
#define KERNEL1_PART 8
#else
#undef KERNEL1_PART
#define KERNEL1_PART 8
#endif // !BLOCK_DIM
	}
	else if (best.kernel1_thread == 8) {
#ifndef KERNEL1_PART
#define KERNEL1_PART 4
#else
#undef KERNEL1_PART
#define KERNEL1_PART 4
#endif // !BLOCK_DIM
	}
	else if (best.kernel1_thread == 16) {
#ifndef KERNEL1_PART
#define KERNEL1_PART 2
#else
#undef KERNEL1_PART
#define KERNEL1_PART 2
#endif // !BLOCK_DIM
	}
	else if (best.kernel1_thread == 32) {
#ifndef KERNEL1_PART
#define KERNEL1_PART 1
#else
#undef KERNEL1_PART
#define KERNEL1_PART 1
#endif // !BLOCK_DIM
	}
}

void assignkernel2_th(Plan best) {
	if (best.kernel2_thread == 2) {
#ifndef KERNEL2_PART
#define KERNEL2_PART 16
#else
#undef KERNEL2_PART
#define KERNEL2_PART 16
#endif // !BLOCK_DIM
	}
	else if (best.kernel2_thread == 4) {
#ifndef KERNEL2_PART
#define KERNEL2_PART 8
#else
#undef KERNEL2_PART
#define KERNEL2_PART 8
#endif // !BLOCK_DIM
	}
	else if (best.kernel2_thread == 8) {
#ifndef KERNEL2_PART
#define KERNEL2_PART 4
#else
#undef KERNEL2_PART
#define KERNEL2_PART 4
#endif // !BLOCK_DIM
	}
	else if (best.kernel2_thread == 16) {
#ifndef KERNEL2_PART
#define KERNEL2_PART 2
#else
#undef KERNEL2_PART
#define KERNEL2_PART 2
#endif // !BLOCK_DIM
	}
	else if (best.kernel2_thread == 32) {
#ifndef KERNEL2_PART
#define KERNEL2_PART 1
#else
#undef KERNEL2_PART
#define KERNEL2_PART 1
#endif // !BLOCK_DIM
	}
}