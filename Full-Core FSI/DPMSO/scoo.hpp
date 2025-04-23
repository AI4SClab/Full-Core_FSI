#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <nmmintrin.h>
#include <map>
#include <vector>
#include <time.h>
#include "config.hpp"

#include "mtx.hpp"
#include <nmmintrin.h>

using namespace std;
/*
//计算32位整数中有多少个1
int _mm_popcnt_u32(int i) {
	i = i - ((i >> 1) & 0x55555555);

	i = (i & 0x33333333) + ((i >> 2) & 0x33333333);

	i = (i + (i >> 4)) & 0x0f0f0f0f;

	i = i + (i >> 8);

	i = i + (i >> 16);

	return i & 0x3f;

}
*/

template<class dataType>
void rearrangecol(MTX<dataType>* mtx, MTX<dataType>* remtx, dataType* x_vector, dataType* revector) {
	// first step sign the number of the non-zero
	int* countarray = new int[mtx->cols];
	memset(countarray, 0, mtx->rows);
	for(int i = 0; i < mtx->nnz; i++){
		countarray[mtx->col[i]]++;
	}

	//second step reorder the cols index
	map<int, vector<int>> m;
	for(int i = 0; i < mtx->cols; i++){
		map<int, vector<int>>::iterator iter = m.find(countarray[i]);
		if (iter != m.end()) {
			iter->second.push_back(i);
		}
		else {
			vector<int> v(1, i);
			m.insert(make_pair(countarray[i], v));
		}
	}
	map<int, vector<int>>::iterator it;
	int* neworder = new int[mtx->cols];
	int ordernum = mtx->cols - 1;
	for(it = m.begin(); it != m.end(); it++){
		vector<int>::iterator it2;
		for (it2 = (*it).second.begin(); it2 != (*it).second.end(); it2++)
		{
			neworder[ordernum--] = (*it2);
		}
    }
	
	//third step sign the sequence of each col
	int* sequence = new int[mtx->cols];
	for(int i = 0; i < mtx->cols; i++){
		sequence[neworder[i]] = i;
	}
	
	//forth step copy the mtx to remtx
	remtx->cols = mtx->cols;
	remtx->rows = mtx->rows;
	remtx->nnz = mtx->nnz;
	remtx->row = (int*)malloc(sizeof(int) * remtx->nnz);
	remtx->col = (int*)malloc(sizeof(int) * remtx->nnz);
	remtx->data = (dataType*)malloc(sizeof(dataType) * remtx->nnz);
	memcpy(remtx->row, mtx->row, sizeof(int) * remtx->nnz);
	memcpy(remtx->data, mtx->data, sizeof(dataType) * remtx->nnz);

	//fifth step revise the col index
	for(int i = 0; i < mtx->nnz; i++){
		remtx->col[i] = sequence[mtx->col[i]];
	}

	//sixth change the mtx to remtx
	sort_coo(remtx);

	//seventh change the vector to revector
	for(int i = 0; i < mtx->cols; i++){
		revector[sequence[i]] = x_vector[i];
	}
	
	delete[] countarray;
	delete[] neworder;
	delete[] sequence;  
}


template<class dataType>
void cpart1tocompress(Cluster_part<dataType>* cpart1) {
	if (cpart1->cluster_blocknum == 0) return;
	int threadnum = 32 * BLOCK_DIM_1;
	int res_entry_num = (cpart1->cluster_blocknum - 1) / threadnum + 2;
	int bitmap_num = (cpart1->cluster_blocknum - 1) / 32 + 1;

	//start by 1,first res_entry is 0,store when i%threadnum == 0 && i!=0
	int res_entry_id = 1;
	// start by 0, store every block 
	int bitmap_id = 0;

	unsigned int* res_entry = new unsigned int[res_entry_num];
	unsigned int* bitmap = new unsigned int[bitmap_num];

	//initialize the bitmap array
	//memset(bitmap, 0, sizeof(unsigned int) * (bitmap_num));
	res_entry[0] = 0;

	unsigned int cur_bitmap = 0;
	int acc = 0;
	int j = 1;
	for (int i = 0; j <= cpart1->cluster_ordernum && i < cpart1->cluster_blocknum; i++) {
		if ((cpart1->order_accumulate_array[j] - 1) == i) {
			int offset = i % 32;
			unsigned int mask = 0x80000000 >> offset;
			cur_bitmap = cur_bitmap | mask;
			j++;
		}
		if (i % threadnum == 0 && i != 0) {
			res_entry[res_entry_id++] = acc;
		}
		if (i % 32 == 31 || i == (cpart1->cluster_blocknum - 1)) {
			bitmap[bitmap_id++] = cur_bitmap;
			acc += _mm_popcnt_u32(cur_bitmap);
			//printf("order_bitmap = %X\n", cur_bitmap);

			// reset the mask and cur_bitmap
			cur_bitmap = 0;
		}
	}
	res_entry[res_entry_id++] = acc;

	cpart1->order_bitmap = new unsigned int[bitmap_id];
	cpart1->order_res_entry = new unsigned int[res_entry_id];
	cpart1->cluster_bitmapnum = bitmap_id;
	cpart1->cluster_resnum = res_entry_id;

	memcpy(cpart1->order_bitmap, bitmap, sizeof(unsigned int) * (bitmap_id));
	memcpy(cpart1->order_res_entry, res_entry, sizeof(unsigned int) * (res_entry_id));

	delete[] res_entry;
	delete[] bitmap;

}

template<class dataType>
void cpart2tocompress(Cluster_part<dataType>* cpart2) {
	if (cpart2->cluster_blocknum == 0) return;
	int threadnum = 32 * BLOCK_DIM_2;
	int res_entry_num = (cpart2->cluster_blocknum - 1) / threadnum + 2;
	int bitmap_num = (cpart2->cluster_blocknum - 1) / 32 + 1;

	//start by 1,first res_entry is 0,store when i%threadnum == 0 && i!=0
	int res_entry_id = 1;
	// start by 0, store every block 
	int bitmap_id = 0;

	unsigned int* res_entry = new unsigned int[res_entry_num];
	unsigned int* bitmap = new unsigned int[bitmap_num];

	//initialize the bitmap array
	//memset(bitmap, 0, sizeof(unsigned int) * (bitmap_num));
	res_entry[0] = 0;

	unsigned int cur_bitmap = 0;
	int acc = 0;
	int j = 1;
	for (int i = 0; j <= cpart2->cluster_ordernum && i < cpart2->cluster_blocknum; i++) {
		if ((cpart2->order_accumulate_array[j] - 1) == i) {
			int offset = i % 32;
			unsigned int mask = 0x80000000 >> offset;
			cur_bitmap = cur_bitmap | mask;
			j++;
		}
		if (i % threadnum == 0 && i != 0) {
			res_entry[res_entry_id++] = acc;
		}
		if (i % 32 == 31 || i == (cpart2->cluster_blocknum - 1)) {
			bitmap[bitmap_id++] = cur_bitmap;
			acc += _mm_popcnt_u32(cur_bitmap);
			//printf("order_bitmap = %X\n", cur_bitmap);

			// reset the mask and cur_bitmap
			cur_bitmap = 0;
		}
	}
	res_entry[res_entry_id++] = acc;

	cpart2->order_bitmap = new unsigned int[bitmap_id];
	cpart2->order_res_entry = new unsigned int[res_entry_id];
	cpart2->cluster_bitmapnum = bitmap_id;
	cpart2->cluster_resnum = res_entry_id;

	memcpy(cpart2->order_bitmap, bitmap, sizeof(unsigned int) * (bitmap_id));
	memcpy(cpart2->order_res_entry, res_entry, sizeof(unsigned int) * (res_entry_id));

	delete[] res_entry;
	delete[] bitmap;

}


template<class dataType>
void sort_dispart(Discrete_part<dataType>* discretepart) {
	int blocknum = 0;
	int nnznum = 0;

	unsigned int* sort_row = new unsigned int[discretepart->discrete_blocknum];
	unsigned int* sort_col = new unsigned int[discretepart->discrete_nnznum];
	unsigned int* sort_index = new unsigned int[discretepart->discrete_blocknum + 1];
	dataType* sort_data = new dataType[discretepart->discrete_nnznum];

	sort_index[0] = 0;
	for (int i = 0; i < discretepart->discrete_blocknum; i++) {
		int blockid = discretepart->sort_blockid_array[i];
		//assignment the accumulate and row array (for block) in discrete part
		sort_row[blocknum++] = discretepart->row_index_array[blockid];

		//assignment the col and data array(for nnz) in discrete part
		for (int j = discretepart->index_accumulate_array[blockid]; j < discretepart->index_accumulate_array[blockid + 1]; j++) {
			sort_col[nnznum] = discretepart->col_index_array[j];
			sort_data[nnznum++] = discretepart->data[j];
		}
		sort_index[blocknum] = nnznum;
	}
	if (nnznum == discretepart->discrete_nnznum && blocknum == discretepart->discrete_blocknum) {
		//cout << "Sort Correct!" << endl;
		memcpy(discretepart->row_index_array, sort_row, sizeof(unsigned int) * blocknum);
		memcpy(discretepart->col_index_array, sort_col, sizeof(unsigned int) * nnznum);
		memcpy(discretepart->index_accumulate_array, sort_index, sizeof(unsigned int) * (blocknum + 1));
		memcpy(discretepart->data, sort_data, sizeof(dataType) * (nnznum));
	}
	delete[] sort_row;
	delete[] sort_col;
	delete[] sort_index;
	delete[] sort_data;
}


template <class dataType>
int mtx2scoo(MTX<dataType>* mtx, int block_length, SCOO<dataType>* scoo) {

	int rows = mtx->rows;
	int cols = mtx->cols;
	int nnz = mtx->nnz;

	scoo->rows = rows;
	scoo->cols = cols;
	scoo->nnz = nnz;

	scoo->block_length = block_length;

	scoo->data = new dataType[scoo->nnz];
	memset(scoo->data, 0, sizeof(dataType) * scoo->nnz);

	//初始化
	unsigned int* all_row_index_array = new unsigned int[nnz];
	unsigned int* all_col_index_array = new unsigned int[nnz];
	unsigned int* all_index_accumulate_array = new unsigned int[nnz + 1];

	//统计块内非零元数量数组，大小为block_length+1
	unsigned int* sumnum = new unsigned int[block_length + 1];
	memset(sumnum, 0, sizeof(unsigned int) * (block_length + 1));

	int i = 0;
	int offset = 0;
	int block_num = 0;
	//bitmap中非零元计数
	int cur_sum = 0;

	int col_num = (mtx->cols + block_length - 1) / block_length;

	//map<int, vector<int>> clus2;

	vector<map<int, vector<int>>> clus(col_num);
	map<int, vector<int>> sort;

	all_index_accumulate_array[0] = 0;

	//pre cluster
	//sign the block whether it has been saved	
	int* flags = new int[nnz];
	memset(flags, 0, sizeof(int) * (nnz));

	//discrete_part array
	int discrete_blocknum = 0;
	int discrete_nnznum = 0;
	unsigned int* discrete_row_array = new unsigned int[nnz];
	unsigned int* discrete_index_array = new unsigned int[nnz + 1];
	unsigned int* discrete_col_array = new unsigned int[nnz];
	unsigned int* discrete_blockid_array = new unsigned int[nnz];

	dataType * discrete_data = new dataType[nnz];

	//initialize three parts array if needed
	discrete_index_array[0] = 0;

	for (i = 0; i < mtx->nnz; i++) {
		all_row_index_array[block_num] = mtx->row[i];
		all_col_index_array[block_num] = mtx->col[i] / block_length * block_length;
		offset = mtx->col[i] - all_col_index_array[block_num];
		int col_id = mtx->col[i] / block_length;

		int cluster_id = offset * block_length;

		cur_sum = 1;

		//未跨行，且在32偏移量内
		while ((i != (mtx->nnz - 1)) && (mtx->row[i + 1] == mtx->row[i]) && ((mtx->col[i + 1] - all_col_index_array[block_num]) < block_length)) {
			i++;
			offset = mtx->col[i] - all_col_index_array[block_num];
			cur_sum++;
		}

		//clusters存储块内元素分布
		cluster_id = cluster_id + offset;

		/*
		//clus2 map<int, vector<int>>
		int sign = col_id * (block_length * block_length) + cluster_id;
		map<int, vector<int>>::iterator iter = clus2.find(sign);
		if (iter != clus2.end()) {
			iter->second.push_back(block_num);
		}
		else {
			vector<int> v(1, block_num);
			clus2.insert(make_pair(sign, v));
		}
		*/

		all_index_accumulate_array[++block_num] = i + 1;

		//data distribution
		sumnum[cur_sum]++;

		//if the block only have two elements
		//clus vector<map<int, vector<int>>> clus;
		if (cur_sum < 3) {
			map<int, vector<int>>::iterator iter = clus[col_id].find(cluster_id);
			if (iter != clus[col_id].end()) {
				iter->second.push_back(block_num - 1);
			}
			else {
				vector<int> v(1, (block_num - 1));
				clus[col_id].insert(make_pair(cluster_id, v));
			}
		}
		else {
			flags[i] = 1;
			/*
			// Sort by nonzero amount in block
			map<int, vector<int>>::iterator iter = sort.find(cur_sum);
			if (iter != sort.end()) {
				iter->second.push_back(discrete_blocknum);
			}
			else {
				vector<int> v(1, discrete_blocknum);
				sort.insert(make_pair(cur_sum, v));
			}
			*/

			//But I find that maybe Sort by colnum can get better results
			//Sort by column, Only change the cur_sum to all_col_index_array[block_num] 
			map<int, vector<int>>::iterator iter = sort.find(all_col_index_array[block_num]);
			if (iter != sort.end()) {
				iter->second.push_back(discrete_blocknum);
			}
			else {
				vector<int> v(1, discrete_blocknum);
				sort.insert(make_pair(all_col_index_array[block_num], v));
			}
			// Sort End

			//assignment the accumulate and row array (for block) in discrete part
			discrete_row_array[discrete_blocknum++] = all_row_index_array[block_num - 1];

			//assignment the col and data array(for nnz) in discrete part
			for (int j = all_index_accumulate_array[block_num - 1]; j < all_index_accumulate_array[block_num]; j++) {
				discrete_col_array[discrete_nnznum] = mtx->col[j];
				discrete_data[discrete_nnznum++] = mtx->data[j];
			}
			discrete_index_array[discrete_blocknum] = discrete_nnznum;
		}

	}
	int sort_blocknum = 0;
	map<int, vector<int>>::iterator sort_it;
	for (sort_it = sort.begin(); sort_it != sort.end(); sort_it++) {
		vector<int>::iterator it2;
		for (it2 = (*sort_it).second.begin(); it2 != (*sort_it).second.end(); it2++)
		{
			discrete_blockid_array[sort_blocknum++] = (*it2);
		}
	}

	/*
	cout << "map size:" << endl << clus2.size()<<endl;
	map<int, vector<int>>::iterator it;
	for (it = clus2.begin(); it != clus2.end(); it++) {
		cout << (*it).first << "\t";
		vector<int>::iterator it2;
		for (it2 = (*it).second.begin(); it2 != (*it).second.end(); it2++)
		{
			cout << *it2 << " ";
		}
		cout << endl;
	}
	*/

	/*
	cout << endl << "data distribution:" << endl;


	//for (i = 1; i < block_length; i++) {
	for (i = 1; i <= 32; i++) {
		cout << "Nonnum:" << i << ",block num" << endl << sumnum[i] << endl;
	}

	float sparsebar = (sumnum[1] + sumnum[2]) * 1.0 / block_num;
	cout << sparsebar << endl;
	*/

	//cluster_part_1 array
	int cpart1_blocknum = 0;
	int cpart1_ordernum = 0;
	unsigned int* cpart1_order = new unsigned int[block_num];
	unsigned int* cpart1_order_array = new unsigned int[block_num + 1];
	unsigned int* cpart1_row_array = new unsigned int[block_num];
	dataType* cpart1_data = new dataType[nnz];

	//cluster_part_2 array
	int cpart2_blocknum = 0;
	int cpart2_ordernum = 0;
	unsigned int* cpart2_order_array = new unsigned int[block_num + 1];
	unsigned int* cpart2_row_array = new unsigned int[block_num];
	unsigned int* cpart2_col_array = new unsigned int[nnz];
	dataType* cpart2_data = new dataType[nnz];

	//initialize three parts array if needed
	cpart1_order_array[0] = 0;
	cpart2_order_array[0] = 0;

	//pre cluster for cpart1 or cpart2
	//vector<map<int, vector<int>>> clus(col_num);
	for (int i = 0; i < col_num; i++) {
		map<int, vector<int>>::iterator it;
		for (it = clus[i].begin(); it != clus[i].end(); it++) {
			//(*it).first cluster id
			int col_basenum = i * block_length;
			int cluster_id = (*it).first;

			//the sign to judge whether there is only one element in the block 
			int cpart1_flag = cluster_id % (block_length + 1);

			//cpart1_flag == 0,the block belongs to the cpart1_flag
			if (cpart1_flag == 0) {

				cpart1_order[cpart1_ordernum] = col_basenum + cluster_id / (block_length + 1);
				vector<int>::iterator it2;
				for (it2 = (*it).second.begin(); it2 != (*it).second.end(); it2++)
				{
					int dataindex = all_index_accumulate_array[(*it2)];
					cpart1_row_array[cpart1_blocknum] = all_row_index_array[(*it2)];
					cpart1_data[cpart1_blocknum++] = mtx->data[dataindex];
				}
				cpart1_order_array[++cpart1_ordernum] = cpart1_blocknum;

			}
			else {
				cpart2_col_array[cpart2_ordernum * 2] = col_basenum + cluster_id / block_length;
				cpart2_col_array[cpart2_ordernum * 2 + 1] = col_basenum + cluster_id % block_length;
				vector<int>::iterator it2;
				for (it2 = (*it).second.begin(); it2 != (*it).second.end(); it2++)
				{
					int dataindex = all_index_accumulate_array[*it2];
					cpart2_row_array[cpart2_blocknum] = all_row_index_array[*it2];
					cpart2_data[cpart2_blocknum * 2] = mtx->data[dataindex];
					cpart2_data[cpart2_blocknum * 2 + 1] = mtx->data[dataindex + 1];
					cpart2_blocknum++;
				}
				cpart2_order_array[++cpart2_ordernum] = cpart2_blocknum;

			}
		}
	}

	scoo->block_num = block_num;

	int up_block_num = ((block_num + 127) >> 7) << 7;
	scoo->up_block_num = up_block_num;
	/*
	//使用packaged_array打包有用信息
	scoo->packaged_array = new unsigned int[4 * up_block_num];
	scoo->array = &scoo->packaged_array[0];
	scoo->index_accumulate_array = &scoo->packaged_array[up_block_num];
	scoo->col_index_array = &scoo->packaged_array[2 * up_block_num];
	scoo->row_index_array = &scoo->packaged_array[3 * up_block_num];
	*/

	scoo->row_index_array = new unsigned int[block_num];
	scoo->col_index_array = new unsigned int[nnz];
	scoo->index_accumulate_array = new unsigned int[block_num + 1];

	memcpy(scoo->index_accumulate_array, all_index_accumulate_array, sizeof(unsigned int) * (block_num + 1));
	memcpy(scoo->row_index_array, all_row_index_array, sizeof(unsigned int) * block_num);
	memcpy(scoo->col_index_array, &(mtx->col[0]), sizeof(unsigned int) * nnz);

	scoo->cluster_part1.cluster_blocknum = cpart1_blocknum;
	scoo->cluster_part1.cluster_ordernum = cpart1_ordernum;
	scoo->cluster_part1.order = new unsigned int[cpart1_ordernum];
	scoo->cluster_part1.order_accumulate_array = new unsigned int[cpart1_ordernum + 1];
	scoo->cluster_part1.row_index_array = new unsigned int[cpart1_blocknum];
	scoo->cluster_part1.data = new dataType[cpart1_blocknum];

	memcpy(scoo->cluster_part1.order, cpart1_order, sizeof(unsigned int) * (cpart1_ordernum));
	memcpy(scoo->cluster_part1.order_accumulate_array, cpart1_order_array, sizeof(unsigned int) * (cpart1_ordernum + 1));
	memcpy(scoo->cluster_part1.row_index_array, cpart1_row_array, sizeof(unsigned int) * (cpart1_blocknum));
	memcpy(scoo->cluster_part1.data, cpart1_data, sizeof(dataType) * (cpart1_blocknum));

	scoo->cluster_part2.cluster_blocknum = cpart2_blocknum;
	scoo->cluster_part2.cluster_ordernum = cpart2_ordernum;
	scoo->cluster_part2.order_accumulate_array = new unsigned int[cpart2_ordernum + 1];
	scoo->cluster_part2.col_index_array = new unsigned int[cpart2_ordernum * 2];
	scoo->cluster_part2.row_index_array = new unsigned int[cpart2_blocknum];
	scoo->cluster_part2.data = new dataType[cpart2_blocknum * 2];

	memcpy(scoo->cluster_part2.order_accumulate_array, cpart2_order_array, sizeof(unsigned int) * (cpart2_ordernum + 1));
	memcpy(scoo->cluster_part2.col_index_array, cpart2_col_array, sizeof(unsigned int) * (cpart2_ordernum * 2));
	memcpy(scoo->cluster_part2.row_index_array, cpart2_row_array, sizeof(unsigned int) * (cpart2_blocknum));
	memcpy(scoo->cluster_part2.data, cpart2_data, sizeof(dataType) * (cpart2_blocknum * 2));

	scoo->discrete_part.discrete_blocknum = discrete_blocknum;
	scoo->discrete_part.discrete_nnznum = discrete_nnznum;
	scoo->discrete_part.row_index_array = new unsigned int[discrete_blocknum];
	scoo->discrete_part.index_accumulate_array = new unsigned int[discrete_blocknum + 1];
	scoo->discrete_part.col_index_array = new unsigned int[discrete_nnznum];
	scoo->discrete_part.sort_blockid_array = new unsigned int[discrete_blocknum];
	scoo->discrete_part.data = new dataType[discrete_nnznum];

	memcpy(scoo->discrete_part.row_index_array, discrete_row_array, sizeof(unsigned int) * (discrete_blocknum));
	memcpy(scoo->discrete_part.index_accumulate_array, discrete_index_array, sizeof(unsigned int) * (discrete_blocknum + 1));
	memcpy(scoo->discrete_part.col_index_array, discrete_col_array, sizeof(unsigned int) * (discrete_nnznum));
	memcpy(scoo->discrete_part.sort_blockid_array, discrete_blockid_array, sizeof(unsigned int)* (discrete_blocknum));
	memcpy(scoo->discrete_part.data, discrete_data, sizeof(dataType) * (discrete_nnznum));


	delete[] all_row_index_array;
	delete[] all_col_index_array;
	delete[] all_index_accumulate_array;
	delete[] sumnum;

	//delete pre cluster array
	delete[] flags;

	//delete cluster_part1 array
	delete[] cpart1_order;
	delete[] cpart1_order_array;
	delete[] cpart1_row_array;
	delete[] cpart1_data;

	//delete cluster_part2 array
	delete[] cpart2_order_array;
	delete[] cpart2_row_array;
	delete[] cpart2_col_array;
	delete[] cpart2_data;

	//delete discrete_part array
	delete[] discrete_row_array;
	delete[] discrete_index_array;
	delete[] discrete_col_array;
	delete[] discrete_blockid_array;
	delete[] discrete_data;

	//scoo->show();

	return scoo->block_num;

}

/* use if judge the discrete_part
//judge the element discrete_part or others
int nnzcount = all_index_accumulate_array[i + 1] - all_index_accumulate_array[i];

if (nnzcount > 2) {
	//assignment the accumulate and row array (for block) in discrete part
	discrete_row_array[discrete_blocknum++] = all_row_index_array[i];

	//assignment the col and data array(for nnz) in discrete part
	for (int j = all_index_accumulate_array[i]; j < all_index_accumulate_array[i + 1]; j++) {
		discrete_col_array[discrete_nnznum] = mtx->col[j];
		discrete_data[discrete_nnznum++] = mtx->data[j];
	}
	discrete_index_array[discrete_blocknum] = discrete_nnznum;
	continue;
}

//use if only one block in the cluster belongs to the discrete_part
//get the size of the cluster to judge which part the block in
map<int, vector<int>>::iterator iter = clus[col_id].find(cluster_id);
int element_num = iter->second.size();

//element_num = 1,the block belongs to the discrete_part
if (element_num == 1) {
	//assignment the accumulate and row array (for block) in discrete part
	discrete_row_array[discrete_blocknum++] = all_row_index_array[i];

	//assignment the col and data array(for nnz) in discrete part
	for (int j = all_index_accumulate_array[i]; j < all_index_accumulate_array[i + 1]; j++) {
		discrete_col_array[discrete_nnznum] = mtx->col[j];
		discrete_data[discrete_nnznum++] = mtx->data[j];
	}
	discrete_index_array[discrete_blocknum] = discrete_nnznum;

	clus[col_id].erase(cluster_id);
}
*/