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
from AuxGeomComp import calculate_distance, split_into_batches, GetBoundDataF,GetBoundDataS

# 设置 MPI 环境
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

    
# def GetBoundDataF(jsonfileF):
#     # 定义最小和最大值
#     min_pre = -56.54605
#     max_pre = 42.18494

#     BoundaryMeshL = []
#     # 从文件中读取字典
#     with open(jsonfileF, "r", encoding="utf-8") as f:
#         ring_dict = json.load(f)
#         print(f"rank{rank}:loadedL中的字典数{len(ring_dict)}....")
#         # for ring_dict in loadedL:
#         for center, boudL in ring_dict.items():
#             FulidD = {}
#             for [ID, X,Y,Z,] in boudL:
#                 pressure = random.uniform(min_pre, max_pre)
#                 node = MeshPoint(ID, X,Y,Z, pressure = pressure)
#                 # FulidL.append(fulp)
#                 FulidD[ID]=node
#             # BoundaryMeshL.append([FulidL,SolidL])
#             BoundaryMeshL.append([FulidD])
#     return BoundaryMeshL


# def GetBoundDataS(jsonfileS,BoundaryMeshLF):
#     # 定义最小和最大值
#     minv = -0.05
#     maxv = 0.05
#     # BoundaryMeshL = []
#     # 从文件中读取字典
#     with open(jsonfileS, "r", encoding="utf-8") as f:
#         ring_dict = json.load(f)
#         print(f"rank{rank}:loadedL中的字典数{len(ring_dict)}....")
#         # for ring_dict in loadedL:
#         i = 0
#         for center, boudL in ring_dict.items():
#             SolidD = {}
#             for [ID, X,Y,Z,] in boudL:
#                 vx = random.uniform(minv, maxv)
#                 vy = random.uniform(minv, maxv)
#                 vz = random.uniform(minv, maxv)
#                 node = MeshPoint(ID, X,Y,Z, vx = vx, vy=vy,vz=vz)
#                 SolidD[ID]=node
#             BoundaryMeshLF[i].append(SolidD)
#             i += 1
#     return BoundaryMeshLF


# def calculate_distance(node1: MeshPoint, node2: MeshPoint) -> float:
#     # 计算两个节点之间的欧几里得距离
#     return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2 + (node1.z - node2.z) ** 2)



def find_k_nearest_neighbors(solidD: Dict[int, MeshPoint], fluidD: Dict[int, MeshPoint], k: int):
    """
    对于 solidD 中的每个固体点，找到 fluidD 中距离它最近的 k 个流体点。
    参数:
        fluidD: 流体点的字典，每个值是 MeshPoint 类型。
        solidD: 固体点的字典，每个值是 MeshPoint 类型。
        k: 最近邻流体点的数量。
    返回:
        List[Tuple[int, List[Tuple[float, int]]]]: 对于每个固体点，返回一个元组，其中包含固体点的 ID 和最近的 k 个流体点及其距离。
    """
    result = []  # 存储每个固体点的最近邻流体点
    for snode in solidD.values():
        heap = []
        for fnode in fluidD.values():
            distance = calculate_distance(snode, fnode)
            # 如果距离为 0，直接返回当前流体点
            if distance == 0:
                result.append((snode.ID, [(0.0, fnode.ID)]))
                break
            if len(heap) < k:
                heapq.heappush(heap, (-distance, fnode.ID))  # 使用负距离，因为 heapq 是最小堆
            else:
                heapq.heappushpop(heap, (-distance, fnode.ID))
        else:
            # 将堆中的结果转换成正距离并存储
            nearest_neighbors = [(-d, fnodeID) for d, fnodeID in heap]
            nearest_neighbors.sort(key=lambda x: x[0])  # 按距离升序排序
            result.append((snode.ID, nearest_neighbors))
    
    return result


def NNI_Compute(solidD, fluidD, nnk): 
    fulid_ret = find_k_nearest_neighbors(solidD, fluidD, nnk)
    for snodeID, fneighbors in fulid_ret:
        # 如果最近邻中有距离为 0 的流体点，直接设置压力值并跳过计算
        if fneighbors[0][0] == 0:
            nearest_fnodeID = fneighbors[0][1]
            solidD[snodeID].pressure = fluidD[nearest_fnodeID].pressure
            continue
        
        # 如果没有距离为 0 的流体点，则进行加权计算
        new_pre = 0.0
        mindist = fneighbors[0][0]  # # 最近邻中最小的距离，假设 fneighbors 非空
        tempd = 0.0
        for dist, fnodeID in fneighbors:
            
            fnode = fluidD[fnodeID]
            if dist == 0:
                print(f"mindist{mindist}, dist={dist}, solidD[snodeID]={solidD[snodeID]},fnode={fnode}")
            new_pre += fnode.pressure * (mindist / dist)
            tempd += (mindist / dist)
        # 更新固体节点 pressure 的值
        solidD[snodeID].pressure = new_pre / tempd if tempd != 0 else 0.0


# # 将 BoundaryMeshL 分成 k 批
# def split_into_batches(BoundaryMeshL, k):
#     n = len(BoundaryMeshL)
#     batch_size = (n + k - 1) // k  # 计算每批次大小
#     return [BoundaryMeshL[i * batch_size : (i + 1) * batch_size] for i in range(k)]


def parallel_NNI_Pressure(batch: List, comm, rank: int, size: int, batch_id: int, nnk: int):
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
        NNI_Compute(solidD, fluidD, nnk)
        
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
    nnk = 2

    #jsonfile = "/work1/wangjue/miaoxue/FSI_Interpolation/resultdict_5mm.json"
    #BoundaryMeshL = GetBoundData(jsonfile)
    #batches = split_into_batches(BoundaryMeshL, k)
    #if rank == 0:
     #   print(f"Total processes: {size}")
     #   print(f"BoundaryMeshL: {len(BoundaryMeshL)}")
     #   print(f"Number of batches: {k}, len(batches): {len(batches)}")
    
     # 主进程处理数据加载
    if rank == 0:
        # 仅主进程加载数据
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

        batch_rets = parallel_NNI_Pressure(batch, comm, rank, size, batch_id, nnk)
        
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
    if rank != 0:
        # 非 Rank 0 的进程将标准输出重定向到 `/dev/null`
        sys.stdout = open(os.devnull, "w")



if __name__ == "__main__":
    # Compute_Pressure()
    test_compute_Pressure()
