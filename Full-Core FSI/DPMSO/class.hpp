#pragma once
#include <iostream>
#include"config.hpp"

using namespace std;

template <class dataType>
struct MTX {
	int rows;
	int cols;
	int nnz;
	int* row;
	int* col;
	dataType* data;

};

//cluster_part1 store the matrix index and data for only one element in the block
//cluster_part2 store the matrix index and data for only two element in the block
template <class dataType>
struct Cluster_part {
	int cluster_blocknum;
	int cluster_ordernum;

	int cluster_bitmapnum;
	int cluster_resnum;

	unsigned int* order;	//commen elements id
	unsigned int* order_accumulate_array;//number of the blocks which has the same order

	unsigned int* order_bitmap;	//bitmap array which stores the col index
	unsigned int* order_res_entry;  //bitmap array which stores one block matrix col

	unsigned int* row_index_array;	//submatrix row number
	unsigned int* col_index_array;  //submatrix col number

	dataType* data;
};


//discrete_part store the matrix index and data for whose element cannot been cluster
template <class dataType>
struct Discrete_part {
	int discrete_blocknum;
	int discrete_nnznum;

	unsigned int* row_index_array;	//submatrix row number
	unsigned int* col_index_array;	//nnz col number
	unsigned int* index_accumulate_array;	//number of the nnz in submatrix 

	unsigned int* sort_blockid_array;

	dataType* data;
};


template <class dataType>
struct SCOO {
	int rows;
	int cols;
	int nnz;

	int block_num;	//submatrix account
	int block_length;	//submatrix slice length

	int up_block_num;	//submatrix number(128倍数)

	unsigned int* row_index_array;	//submatrix row number
	unsigned int* col_index_array;	//nnz col number
	unsigned int* index_accumulate_array;	//number of the nnz in submatrix

	//unsigned int* packaged_array; //packaged of row_index_array,col_index_array,index_accumulate_array

	dataType* data;

	Cluster_part<dataType> cluster_part1;
	Cluster_part<dataType> cluster_part2;
	Discrete_part<dataType> discrete_part;

	void show() {
		cout << "MTX format info: row=\n" << rows << endl << ",cols=\n" << cols << endl << ",nnz=\n" << nnz << endl;
		cout << "SCOO format info: length=" << block_length << ",block_num=" << block_num << endl;
		cout << "Cluster_part only one element: cluster_blocknum = " << cluster_part1.cluster_blocknum;
		cout << ",cluster_ordernum" << cluster_part1.cluster_ordernum << endl;
		cout << "Cluster_part only two element: cluster_blocknum = " << cluster_part2.cluster_blocknum;
		cout << ",cluster_ordernum" << cluster_part2.cluster_ordernum << endl;
		cout << "Discrete part: discrete_blocknum = " << discrete_part.discrete_blocknum;
		cout << ",discrete_nnznum = " << discrete_part.discrete_nnznum << endl;
		
		/*
		cout << "SCOO format info: length=" << block_length << ",block_num=" << block_num << endl;
		for (int i = 0; i < block_num; i++) {
			cout << "num:" << i << "  row:" << row_index_array[i];
			cout << "	index_accumulate_array:" << index_accumulate_array[i] << "  bitmap	cols_num:";
			int j = 0;
			if (i == block_num - 1) {
				for (j = index_accumulate_array[i]; j < nnz; j++) {
					cout << col_index_array[j] << "\t";
				}
			}
			else {
				for (j = index_accumulate_array[i]; j < index_accumulate_array[i + 1]; j++) {
					cout << col_index_array[j] << "\t";
				}
			}
			cout << endl;
			if (i == block_num - 1)
				cout << "num:" << i + 1 << "	index_accumulate_array:" << index_accumulate_array[i + 1] << endl;
		}
		
		cout << "Cluster_part only one element: cluster_blocknum = " << cluster_part1.cluster_blocknum;
		cout << ",cluster_ordernum" << cluster_part1.cluster_ordernum << endl;
		if (cluster_part1.order_accumulate_array[cluster_part1.cluster_ordernum] == cluster_part1.cluster_blocknum)
			cout << "nnz num correct!" << endl;
		for (int i = 0; i < cluster_part1.cluster_ordernum; i++) {
			cout << "cluster order = " << cluster_part1.order[i] << endl;
			for (int j = cluster_part1.order_accumulate_array[i]; j < cluster_part1.order_accumulate_array[i + 1]; j++) {
				cout << "num = " << j << ",cluster row = " << cluster_part1.row_index_array[j];
				cout << ",cluster data = " << cluster_part1.data[j];
				cout << endl;
			}
		}
		
		// cpart1 bit array information
		for (int i = 0; i < cluster_part1.cluster_bitmapnum; i++) {
			cout << "num:" << i;
			printf("order_bitmap = %X\n", cluster_part1.order_bitmap[i]);
		}
		for (int i = 0; i < cluster_part1.cluster_resnum; i++) {
			cout << "num:" << i << "res_entry = " << cluster_part1.order_res_entry[i] << endl;
		}
		
		cout << "Cluster_part only two element: cluster_blocknum = " << cluster_part2.cluster_blocknum;
		cout << ",cluster_ordernum" << cluster_part2.cluster_ordernum << endl;
		if (cluster_part2.order_accumulate_array[cluster_part2.cluster_ordernum] == cluster_part2.cluster_blocknum)
			cout << "nnz num correct!" << endl;
		for (int i = 0; i < cluster_part2.cluster_ordernum; i++) {
			cout << "cluster col = " << cluster_part2.col_index_array[i * 2] << "\t"<<cluster_part2.col_index_array[i * 2 + 1]<<endl;
			for (int j = cluster_part2.order_accumulate_array[i]; j < cluster_part2.order_accumulate_array[i + 1]; j++) {
				cout << "num = " << j << ",cluster row = " << cluster_part2.row_index_array[j];
				cout << ",cluster data = " << cluster_part2.data[j * 2] << " " << cluster_part2.data[j * 2 + 1];
				cout << endl;
			}
		}
		
		// cpart2 bit array information
		for (int i = 0; i < cluster_part2.cluster_bitmapnum; i++) {
			cout << "num:" << i;
			printf("order_bitmap = %X\n", cluster_part2.order_bitmap[i]);
		}
		for (int i = 0; i < cluster_part2.cluster_resnum; i++) {
			cout << "num:" << i << "res_entry = " << cluster_part2.order_res_entry[i] << endl;
		}
		
		cout << "Discrete part: discrete_blocknum = " << discrete_part.discrete_blocknum;
		cout << ",discrete_nnznum = " << discrete_part.discrete_nnznum << endl;
		if (discrete_part.index_accumulate_array[discrete_part.discrete_blocknum] == discrete_part.discrete_nnznum)
			cout << "nnz num correct!" << endl;
		for (int i = 0; i < discrete_part.discrete_blocknum; i++) {
			cout << "num:" << i << "  row:" << discrete_part.row_index_array[i];
			cout << "	index_accumulate_array:" << discrete_part.index_accumulate_array[i] << endl;
			for (int j = discrete_part.index_accumulate_array[i]; j < discrete_part.index_accumulate_array[i + 1]; j++) {
				cout << "num = " << j << ",col = " << discrete_part.col_index_array[j];
				cout << ",data = " << discrete_part.data[j];
				cout << endl;
			}
		}
		*/
	}
};

template <class dataType>
void calstorage(SCOO<dataType> sc, Plan best) {
	int mtx_storage = sc.nnz * 2 * sizeof(int) + sc.nnz * sizeof(dataType);
	int csr_storage = (sc.rows + 1 + sc.nnz) * sizeof(int) + sc.nnz * sizeof(dataType);
	int scoo_storage = sc.nnz * sizeof(dataType);
	if (best.kernel1 == 0) {
		scoo_storage += (sc.cluster_part1.cluster_blocknum + sc.cluster_part1.cluster_ordernum * 2 + 1) * sizeof(unsigned int);
	}
	else {
		scoo_storage += (sc.cluster_part1.cluster_blocknum + sc.cluster_part1.cluster_ordernum + 
			sc.cluster_part1.cluster_bitmapnum + sc.cluster_part1.cluster_resnum) * sizeof(unsigned int);
	}

	if (best.kernel2 == 0) {
		scoo_storage += (sc.cluster_part2.cluster_blocknum + sc.cluster_part2.cluster_ordernum * 3 + 1) * sizeof(unsigned int);
	}
	else {
		scoo_storage += (sc.cluster_part2.cluster_blocknum + sc.cluster_part2.cluster_ordernum * 2 +
			sc.cluster_part2.cluster_bitmapnum + sc.cluster_part2.cluster_resnum) * sizeof(unsigned int);
	}

	scoo_storage += (sc.discrete_part.discrete_blocknum * 2 + 1 + sc.discrete_part.discrete_nnznum) * sizeof(unsigned int);

	cout << "MTX storage =\n" << mtx_storage << endl;
	cout << "CSR storage =\n" << csr_storage << endl;
	cout << "SCOO storage =\n" << scoo_storage << endl;
}
