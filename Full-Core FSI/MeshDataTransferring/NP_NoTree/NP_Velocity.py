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
from AuxGeomComp import is_point_in_quadrilateral,quadrilateral_barycentric_interpolation_velocity,is_point_on_edge,linear_interpolation_velocity,find_k_nearest_neighbors, GetBoundFaceData, GetBoundDataF,GetBoundDataS,calculate_distance,split_into_batches

# 设置 MPI 环境
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



def NP_Compute(fluidD, solidD, boundface_dict, nnk=1): 
    Solid_ret = find_k_nearest_neighbors(fluidD, solidD,  nnk)  # 寻找最近的固体点
    
    for (fnodeID, [(mindist, nearest_snodeID)]) in Solid_ret:
        fulid_node = fluidD[fnodeID]
        # print(f"插值计算前：fulid_node = {fulid_node}")
        # 如果距离为0，则直接赋值
        if mindist == 0:
            # print("距离为0，直接赋值........")
            fulid_node.vx = solidD[nearest_snodeID].vx  # 最近点的速度直接赋值
            fulid_node.vy = solidD[nearest_snodeID].vy  
            fulid_node.vz = solidD[nearest_snodeID].vz  
        else:
            # 找到最近固体点关联的四边形
            s_faceL = boundface_dict[str(nearest_snodeID)]
            # print(f"距离非0={mindist}, len(f_faceL) = {len(f_faceL)}......")
            for [sid1, sid2, sid3, sid4] in s_faceL:
                # print(f"[fid1, fid2, fid3, fid4] = {[fid1, fid2, fid3, fid4]}")
                sn1 = solidD[sid1]
                sn2 = solidD[sid2]
                sn3 = solidD[sid3]
                sn4 = solidD[sid4]
                # 判断 fulid_node 是否在四边形内部 
                if is_point_in_quadrilateral(fulid_node, sn1, sn2, sn3, sn4):
                    # print(f"距离非0={mindist}，在关联四边形{[fn1, fn2, fn3, fn4]}内........")
                    # 如果在内部，进行四边形的重心插值
                    vx,vy,vz = quadrilateral_barycentric_interpolation_velocity(
                        fulid_node, sn1, sn2, sn3, sn4
                    )
                    fulid_node.vx = vx
                    fulid_node.vy = vy
                    fulid_node.vz = vz
                    # print(f"距离非0={mindist}，在关联四边形{[sn1, sn2, sn3, sn4]}内........{fulid_node.vx, fulid_node.vy,fulid_node.vz}")
                    break  # 找到一个四边形即可退出循环
                
                elif is_point_on_edge(fulid_node, sn1, sn2):
                    # 如果在边上，进行线性插值
                    vx,vy,vz = linear_interpolation_velocity(fulid_node, sn1, sn2)
                    fulid_node.vx = vx
                    fulid_node.vy = vy
                    fulid_node.vz = vz
                    # print(f"1距离非0={mindist}，在关联四边形{[sn1, sn2, sn3, sn4]}的边{[sn1, sn2]}上........压力值={fulid_node}")
                    break
                elif is_point_on_edge(fulid_node, sn2, sn3):
                    vx,vy,vz = linear_interpolation_velocity(fulid_node, sn2, sn3)
                    fulid_node.vx = vx
                    fulid_node.vy = vy
                    fulid_node.vz = vz
                    # print(f"2距离非0={mindist}，在关联四边形{[sn1, sn2, sn3, sn4]}的边{[sn2, sn3]}上........压力值={fulid_node}")
                    break
                elif is_point_on_edge(fulid_node, sn3, sn4):
                    vx,vy,vz = linear_interpolation_velocity(fulid_node, sn3, sn4)
                    fulid_node.vx = vx
                    fulid_node.vy = vy
                    fulid_node.vz = vz
                    # print(f"3距离非0={mindist}，在关联四边形{[sn1, sn2, sn3, sn4]}的边{[sn3, sn4]}上........压力值={fulid_node}")
                    break
                elif is_point_on_edge(fulid_node, sn4, sn1):
                    vx,vy,vz = linear_interpolation_velocity(fulid_node, sn4, sn1)
                    fulid_node.vx = vx
                    fulid_node.vy = vy
                    fulid_node.vz = vz
                    # print(f"4距离非0={mindist}，在关联四边形{[sn1, sn2, sn3, sn4]}的边{[sn4, sn1]}上........压力值={fulid_node}")
                    break


def parallel_NP_Velocity(batch: List, boundface_dict:Dict, comm, rank: int, size: int, batch_id: int, nnk: int):
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
        NP_Compute(fluidD, solidD, boundface_dict, nnk)
        
        local_ret.append((f"batch_{batch_id}_{i}_Fulid", solidD))
    
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
    

def Compute_Velocity():
    """
    该函数读取输入数据，划分为多个批次，
    依次对每个批次调用 parallel_NP_Velocity 进行处理，
    处理完一个批次后将结果保存到磁盘，并更新索引；
    最终 rank 0 进程保存所有批次存入文件的索引信息并返回索引。
    """
    k = 8
    nnk = 1
    
     # 主进程处理数据加载
    if rank == 0:
        # 仅主进程加载数据
        boundface_dict = GetBoundFaceData("/work1/wangjue/miaoxue/FSI_Interpolation/jsondata/facedict_10_P.json")
        BoundaryMeshLF = GetBoundDataF("/work1/wangjue/miaoxue/FSI_Interpolation/jsondata/ringdict_10_F.json")
        BoundaryMeshL = GetBoundDataS("/work1/wangjue/miaoxue/FSI_Interpolation/jsondata/ringdict_10_P.json", BoundaryMeshLF)
    
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

        batch_rets = parallel_NP_Velocity(batch, boundface_dict, comm, rank, size, batch_id, nnk)
        
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


def test_compute_Velocity():
    # 仅在 Rank 0 汇总时间和内存数据作为示例
    if rank == 0:
        print("Starting Compute_Velocity test...")

    tracemalloc.start()  # 开始内存追踪
    start_time = time.time()
    
    # 调用并行计算函数，注意各进程都需要调用 Compute_Velocity
    forest_index = Compute_Velocity()
    
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
    test_compute_Velocity()
