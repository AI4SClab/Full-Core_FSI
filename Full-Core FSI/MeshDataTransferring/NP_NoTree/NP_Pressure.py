from mpi4py import MPI
import random, json, math
from typing import Dict, List, Tuple
import heapq
import tracemalloc,time
import sys
import os

# 获取父目录路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将父目录添加到 sys.path
sys.path.append(parent_dir)

from TreeClass import MeshPoint
from saveandloadFuns import save_forest_index, save_batch_to_disk
from AuxGeomComp import is_point_in_quadrilateral,quadrilateral_barycentric_interpolation,is_point_on_edge,linear_interpolation,find_k_nearest_neighbors, GetBoundFaceData, GetBoundDataF,GetBoundDataS,split_into_batches


# 设置 MPI 环境
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()





def NP_Compute(solidD, fluidD, boundface_dict, nnk=1): 
    fluid_ret = find_k_nearest_neighbors(solidD, fluidD, nnk)  # 寻找最近的流体点
    
    for (snodeID, [(mindist, nearest_fnodeID)]) in fluid_ret:
        solid_node = solidD[snodeID]
        # print(f"插值计算前：solid_node = {solid_node}")
        # 如果距离为0，则直接赋值
        if mindist == 0:
            # print("距离为0，直接赋值........")
            solid_node.pressure = fluidD[nearest_fnodeID].pressure  # 最近点的压力直接赋值
        else:
            # 找到最近流体点关联的四边形
            f_faceL = boundface_dict[str(nearest_fnodeID)]
            # print(f"距离非0={mindist}, len(f_faceL) = {len(f_faceL)}......")
            for [fid1, fid2, fid3, fid4] in f_faceL:
                # print(f"[fid1, fid2, fid3, fid4] = {[fid1, fid2, fid3, fid4]}")
                fn1 = fluidD[fid1]
                fn2 = fluidD[fid2]
                fn3 = fluidD[fid3]
                fn4 = fluidD[fid4]
                # 判断 solid_node 是否在四边形内部 
                if is_point_in_quadrilateral(solid_node, fn1, fn2, fn3, fn4):
                    # print(f"距离非0={mindist}，在关联四边形{[fn1, fn2, fn3, fn4]}内........")
                    # 如果在内部，进行四边形的重心插值
                    solid_node.pressure = quadrilateral_barycentric_interpolation(
                        solid_node, fn1, fn2, fn3, fn4
                    )
                    # print(f"距离非0={mindist}，在关联四边形{[fn1, fn2, fn3, fn4]}内........{solid_node.pressure}")
                    break  # 找到一个四边形即可退出循环
                
                elif is_point_on_edge(solid_node, fn1, fn2):
                    # 如果在边上，进行线性插值
                    solid_node.pressure = linear_interpolation(solid_node, fn1, fn2)
                    # print(f"1距离非0={mindist}，在关联四边形{[fn1, fn2, fn3, fn4]}的边{[fn1, fn2]}上........压力值={solid_node.pressure}")
                    break
                elif is_point_on_edge(solid_node, fn2, fn3):
                    solid_node.pressure = linear_interpolation(solid_node, fn2, fn3)
                    # print(f"2距离非0={mindist}，在关联四边形{[fn1, fn2, fn3, fn4]}的边{[fn2, fn3]}上........压力值={solid_node.pressure}")
                    break
                elif is_point_on_edge(solid_node, fn3, fn4):
                    solid_node.pressure = linear_interpolation(solid_node, fn3, fn4)
                    # print(f"3距离非0={mindist}，在关联四边形{[fn1, fn2, fn3, fn4]}的边{[fn3, fn4]}上........压力值={solid_node.pressure}")
                    break
                elif is_point_on_edge(solid_node, fn4, fn1):
                    solid_node.pressure = linear_interpolation(solid_node, fn4, fn1)
                    # print(f"4距离非0={mindist}，在关联四边形{[fn1, fn2, fn3, fn4]}的边{[fn4, fn1]}上........压力值={solid_node.pressure}")
                    break



def parallel_NP_Pressure(batch: List, boundface_dict:Dict, comm, rank: int, size: int, batch_id: int, nnk: int):
    """
    在每个进程中对当前批次进行处理。
    对于批次中的每一组 [fluidD, solidD]，调用 find_k_nearest_neighbors，
    并利用最近邻信息更新 solidD 中各节点的 pressure 值；
    然后将每个进程的结果通过 MPI.gather 收集到 rank 0 进程，
    rank 0 将各个进程的局部结果整合成一个 batch_forest 并返回。
    """
    local_ret = []
    for i, [fluidD, solidD] in enumerate(batch):
        # 按照 i % size 进行分配，这里 size 为进程数，假设：由于 batch 总共有 8 个任务，而 size=32，只有当 i < 8 且 i % 32 == rank 的进程进行计算
        if i % size != rank:
            continue  # 其它进程不处理该任务，避免资源浪费
        NP_Compute(solidD, fluidD, boundface_dict, nnk)
        
        local_ret.append((f"batch_{batch_id}_{i}_Solid", solidD))
    
    # 同步所有进程后收集结果
    comm.Barrier()
    all_rets = comm.gather(local_ret, root=0)
    
    if rank == 0:
        batch_forest = {}
        for process_trees in all_rets:
            for key, tree_data in process_trees:
                batch_forest[key] = tree_data

        # 调试输出部分结果
        print(f"[DEBUG] Batch {batch_id} forest keys: {list(batch_forest.keys())[:10]}")
        print(f"[DEBUG] Batch {batch_id} forest size: {len(batch_forest)}")
        return batch_forest
    else:
        return None
    



def Compute_Pressure():
    """
    该函数读取输入数据，划分为多个批次，
    依次对每个批次调用 parallel_NNI_Pressure 进行处理，
    处理完一个批次后将结果保存到磁盘，并更新索引；
    最终 rank 0 进程保存所有批次存入文件的索引信息并返回索引。
    """
    k = 1
    nnk = 1
    
     # 主进程处理数据加载
    if rank == 0:
        # 仅主进程加载数据
        
        boundface_dict = GetBoundFaceData("D:/Code/FSI_Interpolation2/jsondata/1.6W-12.6W/1facedict_1350_F_12.6W.json")
        # BoundaryMeshLF = GetBoundDataF("D:\\Code\\a800\\new\\FSI_Interpolation\\jsondata\\ringdict_10_F.json")
        # BoundaryMeshL = GetBoundDataS("D:\\Code\\a800\\new\\FSI_Interpolation\\jsondata\\ringdict_10_P.json", BoundaryMeshLF)
        BoundaryMeshLF = GetBoundDataF("D:/Code/FSI_Interpolation2/jsondata/1.6W-12.6W/1ringdict_1350_F_12.6W.json")
        BoundaryMeshL = GetBoundDataS("D:/Code/FSI_Interpolation2/jsondata/1.6W-12.6W/1ringdict_1350_P_1.6W.json", BoundaryMeshLF)
        print(f"rank{rank}:需要构建的流体树及固体树总数: {len(BoundaryMeshL)*2}")
        
        # 将 BoundaryMeshL 按批次划分
        batches = split_into_batches(BoundaryMeshL, k)

        print(f"Total processes: {size}")
        print(f"BoundaryMeshL: {len(BoundaryMeshL)}")
        print(f"Number of batches: {k}, len(batches): {len(batches)}")
    else:
        # 非主进程初始化为空
        batches = None
        boundface_dict = None

    # 将 batches 广播给所有进程
    batches = comm.bcast(batches, root=0)
    boundface_dict = comm.bcast(boundface_dict, root=0)

    # 确保所有进程已经接收到 batches
    if batches is None or boundface_dict is None:
        raise ValueError("Batches or boundface_dict not properly broadcasted!")

    index = {}  # 所有批次文件存储索引
    for batch_id, batch in enumerate(batches):
        if rank == 0:
            print(f"\nProcessing batch {batch_id + 1}/{len(batches)}，当前批次中 [fluidD, solidD] 对象数量：{len(batch)}")
        # 同步：确保所有进程在开始新的批次前处于同一状态
        comm.Barrier()

        batch_rets = parallel_NP_Pressure(batch, boundface_dict, comm, rank, size, batch_id, nnk)
        
        if rank == 0:
            if batch_rets:
                print(f"[DEBUG] Batch {batch_id} 包含 {len(batch_rets)} 个结果.")
            else:
                print(f"[DEBUG] Batch {batch_id} forest is empty!")
            
            # 保存当前批次的结果到磁盘
            file_path = save_batch_to_disk(batch_rets, batch_id)
            index[batch_id] = file_path # 将当前批次的存储信息存入index中

        # 同步：确保每个批次的存盘完成后再进入下一个批次
        comm.Barrier()

    if rank == 0:
        print(f"\n[DEBUG] Forest index contains {len(index)} entries.")
        for bid, file_path in index.items():
            print(f"  Batch {bid} stored at: {file_path}")
        save_forest_index(index)
        print(f"Total batches saved: {len(index)}")
        return index
    else:
        return None


def test_compute_Pressure():
    # 仅在 Rank 0 汇总时间和内存数据作为示例
    if rank == 0:
        print("Starting Compute_Pressure test...")

    tracemalloc.start()  # 开始内存追踪
    start_time = time.time()
    
    # 调用并行计算函数，注意各进程都需要调用 Compute_Velocity
    forest_index = Compute_Pressure()
    
    # Barrier 确保所有进程计算完毕
    comm.Barrier()
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 这里只在 Rank 0 输出整体统计信息
    if rank == 0:
        print("\n=== Test Performance Report ===")
        print(f"Total Running Time: {end_time - start_time:.4f} seconds")
        print(f"Current Memory Usage: {current/1024**2:.4f} MB")
        print(f"Peak Memory Usage: {peak/1024**2:.4f} MB")
        print("Forest Index:", forest_index)


if __name__ == "__main__":
    # Compute_Pressure()
    test_compute_Pressure()
