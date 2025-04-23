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
from RBFCompute import compute_pressure
from AuxGeomComp import split_into_batches

# 设置 MPI 环境
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

    
def GetBoundDataF(jsonfileF, boundnodesdisp):
    # 定义最小和最大值
    min_pre = -56.54605
    max_pre = 42.18494

    BoundaryMeshL = []
    # 从文件中读取字典
    with open(jsonfileF, "r", encoding="utf-8") as f:
        ring_dict = json.load(f)
        print(f"rank{rank}:loadedL中的字典数{len(ring_dict)}....")
        for center, boudL in ring_dict.items():
            FulidL =[]
            for [ID, X,Y,Z,] in boudL:
                pressure = random.uniform(min_pre, max_pre)
                # node = MeshPoint(ID, X,Y,Z, pressure = pressure)
               # pressure = boundnodesdisp[str(ID)]["toatal_disp"]
                node = MeshPoint(ID, X,Y,Z, pressure = pressure)
                FulidL.append(node)
            BoundaryMeshL.append([FulidL])
            
    return BoundaryMeshL


def GetBoundDataS(jsonfileS,BoundaryMeshLF):
    # 定义最小和最大值
    minv = -0.05
    maxv = 0.05
    # BoundaryMeshL = []
    # 从文件中读取字典
    with open(jsonfileS, "r", encoding="utf-8") as f:
        ring_dict = json.load(f)
        print(f"rank{rank}:loadedL中的字典数{len(ring_dict)}....")
        # for ring_dict in loadedL:
        i = 0
        for center, boudL in ring_dict.items():
            SolidL = []
            for [ID, X,Y,Z,] in boudL:
                vx = random.uniform(minv, maxv)
                vy = random.uniform(minv, maxv)
                vz = random.uniform(minv, maxv)
                node = MeshPoint(ID, X,Y,Z, vx = vx, vy=vy,vz=vz)
                SolidL.append(node)
            BoundaryMeshLF[i].append(SolidL)
            i += 1
    return BoundaryMeshLF


# def calculate_distance(node1: MeshPoint, node2: MeshPoint) -> float:
#     # 计算两个节点之间的欧几里得距离
#     return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2 + (node1.z - node2.z) ** 2)



def RBF_Pressure_Compute(solidL, fluidL, R): 
    pressure_matrix_s = compute_pressure( solidL, fluidL, R)  # 为固体网格点计算压力值
    # print(f"len(solid_points), len(fluid_points), pressure_matrix_s.shape:{len(solid_points), len(fluid_points), pressure_matrix_s.shape}")
    # 更新固体结点的压力值
    for i in range(len(solidL)):
        solidL[i].pressure = pressure_matrix_s[i,0] # 更新各个固体网格点的压力值


# # 将 BoundaryMeshL 分成 k 批
# def split_into_batches(BoundaryMeshL, k):
#     n = len(BoundaryMeshL)
#     batch_size = (n + k - 1) // k  # 计算每批次大小
#     return [BoundaryMeshL[i * batch_size : (i + 1) * batch_size] for i in range(k)]


def parallel_RBF_Pressure(batch: List, comm, rank: int, size: int, batch_id: int, R:float):
    """
    在每个进程中对当前批次进行处理。
    对于批次中的每一组 [fluidD, solidD]，调用 find_k_nearest_neighbors，
    并利用最近邻信息更新 solidD 中各节点的 pressure 值；
    然后将每个进程的结果通过 MPI.gather 收集到 rank 0 进程，
    rank 0 将各个进程的局部结果整合成一个 batch_forest 并返回。
    """
    local_ret = []
    for i, [fluidL, solidL] in enumerate(batch):
        # 按照 i % size 进行分配，这里 size 为进程数，假设：由于 batch 总共有 8 个任务，而 size=32，只有当 i < 8 且 i % 32 == rank 的进程进行计算
        if i % size != rank:
            continue  # 其它进程不处理该任务，避免资源浪费
        RBF_Pressure_Compute(solidL, fluidL, R)
        
        local_ret.append((f"batch_{batch_id}_{i}_Solid", solidL))
    
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
    
def GetBoundFaceData(jsonfile):
    boundface_dict = {}
    # 从文件中读取字典
    with open(jsonfile, "r", encoding="utf-8") as f:
        boundface_dict = json.load(f)
    return boundface_dict
    

def Compute_Pressure():
    """
    该函数读取输入数据，划分为多个批次，
    依次对每个批次调用 parallel_NNI_Pressure 进行处理，
    处理完一个批次后将结果保存到磁盘，并更新索引；
    最终 rank 0 进程保存所有批次存入文件的索引信息并返回索引。
    """
    k = 1
    R = 5

    #jsonfile = "/work1/wangjue/miaoxue/FSI_Interpolation/resultdict_5mm.json"
    #BoundaryMeshL = GetBoundData(jsonfile)
    #batches = split_into_batches(BoundaryMeshL, k)
    #if rank == 0:
     #   print(f"Total processes: {size}")
     #   print(f"BoundaryMeshL: {len(BoundaryMeshL)}")
     #   print(f"Number of batches: {k}, len(batches): {len(batches)}")
    
     # 主进程处理数据加载
    if rank == 0:
        boundnodesdisp =[] #GetBoundFaceData("/work1/wangjue/miaoxue/FSI_Interpolation/jsondata/fulid_pressure_org.json")
        # 仅主进程加载数据
        BoundaryMeshLF = GetBoundDataF("D:/Code/FSI_Interpolation2/jsondata/1.6W-12.6W/1ringdict_1350_F_12.6W.json", boundnodesdisp)
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

    # 将 batches 广播给所有进程
    batches = comm.bcast(batches, root=0)

    # 确保所有进程已经接收到 batches
    if batches is None:
        raise ValueError("Batches not properly broadcasted!")

    index = {}  # 所有批次文件存储索引
    for batch_id, batch in enumerate(batches):
        if rank == 0:
            print(f"\nProcessing batch {batch_id + 1}/{len(batches)}，当前批次中 [fluidD, solidD] 对象数量：{len(batch)}")
        # 同步：确保所有进程在开始新的批次前处于同一状态
        comm.Barrier()

        batch_rets = parallel_RBF_Pressure(batch, comm, rank, size, batch_id, R)
        
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

    # if rank == 0:
    #     print(f"\n[DEBUG] Forest index contains {len(index)} entries.")
    #     for bid, file_path in index.items():
    #         print(f"  Batch {bid} stored at: {file_path}")
    #     save_forest_index(index)
    #     print(f"Total batches saved: {len(index)}")
    #     return index
    # else:
    #     return None


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
