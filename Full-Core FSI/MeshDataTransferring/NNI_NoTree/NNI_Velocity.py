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


# def GetBoundData(jsonfile):
#     # 定义最小和最大值
#     min_pre = -56.54605
#     max_pre = 42.18494
#     minv = -0.05
#     maxv = 0.05
#     BoundaryMeshL = []
#     # 从文件中读取字典
#     with open(jsonfile, "r", encoding="utf-8") as f:
#         loadedL = json.load(f)
#         print(f"loadedL中的字典数{len(loadedL)}....")
#         for ring_dict in loadedL:
#             for center, boudL in ring_dict.items():
#                 # FulidL = []
#                 # SolidL = []
#                 FulidD = {}
#                 SolidD = {}
#                 for [ID, X,Y,Z,] in boudL:
#                     pressure = random.uniform(min_pre, max_pre)
#                     vx = random.uniform(minv, maxv)
#                     vy = random.uniform(minv, maxv)
#                     vz = random.uniform(minv, maxv)
#                     fulp = MeshPoint(ID, X,Y,Z, pressure = pressure)
#                     solp = MeshPoint(ID, X+random.uniform(-0.1, 0.1),Y+random.uniform(-0.1, 0.1),Z+random.uniform(-0.1, 0.1), vx=vx, vy=vy, vz=vz)
#                     FulidD[ID] = fulp
#                     SolidD[ID] = solp
#                 BoundaryMeshL.append([FulidD,SolidD])
#     # print(f"需要构建的流体树及固体树总数: {len(BoundaryMeshL)*2}")
#     return BoundaryMeshL

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

# # 将 BoundaryMeshL 分成 k 批
# def split_into_batches(BoundaryMeshL, k):
#     n = len(BoundaryMeshL)
#     batch_size = (n + k - 1) // k  # 计算每批次大小
#     return [BoundaryMeshL[i * batch_size : (i + 1) * batch_size] for i in range(k)]


def find_k_nearest_neighbors(fluidD: Dict[int, MeshPoint], solidD: Dict[int, MeshPoint], k: int):# -> List[List[Tuple[MeshPoint, float]]]:
    """
    对于fluidL中的每个流体点，找到solidL中距离它最近的k个固体点。
    参数:
        fluidL: 流体点的列表，每个元素是Node类型。
        solidL: 固体点的列表，每个元素是Node类型。
        k: 最近邻固体点的数量。
    
    返回:List[List[Tuple[Node, float]]]: 对于每个流体点，返回一个列表，其中包含k个最近邻固体点及其距离。
    """
    result = []  # 存储每个流体点的k个最近邻固体点
    for fnode in fluidD.values():
        # 使用堆来找到最近的k个点
        heap = []
        for snode in solidD.values():
            # 计算流体点与固体点之间的距离
            distance = calculate_distance(fnode, snode)
            # 如果距离为 0，直接返回当前流体点
            if distance == 0:
                result.append((fnode.ID, [(0.0, snode.ID)]))
                break
            
            # 将距离和固体点加入堆中
            if len(heap) < k:
                heapq.heappush(heap, (-distance, snode.ID))  # 使用负距离，因为heapq是最小堆
            else:
                # 如果堆的长度已经是k，则只保留更近的点
                heapq.heappushpop(heap, (-distance, snode.ID))
        else:
            # 将堆中的结果转换成正距离并存储
            nearest_neighbors = [(-d, snodeID) for d, snodeID in heap]
            nearest_neighbors.sort(key=lambda x: x[0])  # 按距离升序排序
            result.append([fnode.ID, nearest_neighbors])
    
    return result


def parallel_NNI_Velocity(batch, comm, rank, size, batch_id, nnk):
    """
    并行处理一个批次数据：
      对于批次中的每一对 [fluidD, solidD]，
      1. 利用 solidD 中的节点作为近邻，更新 fluidD 中各节点的速度值，
         更新方法为：利用固体节点的速度加权平均插值。
      2. 将每组更新后的 fluidD 结果保存到局部列表 local_ret 中，
         格式为 (key, fluidD) 形式，其中 key 为批次及元素标识。
      3. 使用 MPI.gather 将所有进程的结果送到 root 进程进行整合。
    """
    local_ret = []
    for i, [fluidD, solidD] in enumerate(batch):
        # 按照 i % size 进行分配，这里 size 为进程数，假设：由于 batch 总共有 8 个任务，而 size=32，只有当 i < 8 且 i % 32 == rank 的进程进行计算
        if i % size != rank:
            continue  # 其它进程不处理该任务，避免资源浪费
        # 这里调用近邻查找函数，参数顺序：对 fluidD 中的每个节点在固体节点中寻找最近邻
        solid_ret = find_k_nearest_neighbors(fluidD, solidD, nnk)
        for fnodeID, sneighbors in solid_ret:
            # 如果最近邻中有距离为 0 的固体点，直接设置速度值并跳过计算
            if sneighbors[0][0] == 0:
                nearest_snodeID = sneighbors[0][1]
                fluidD[fnodeID].vx = solidD[nearest_snodeID].vx
                fluidD[fnodeID].vy = solidD[nearest_snodeID].vy
                fluidD[fnodeID].vz = solidD[nearest_snodeID].vz
                continue

            new_vx = 0.0
            new_vy = 0.0
            new_vz = 0.0
            # 采用第一个邻居的距离作为基准
            mindist = sneighbors[0][0]
            tempd = 0.0
            for dist, snodeID in sneighbors:
                snode = solidD[snodeID]
                # print(f"  固体点 {snode} 距离: {dist:.2f}")
                new_vx += snode.vx * (mindist/dist)
                new_vy += snode.vy * (mindist/dist)
                new_vz += snode.vz * (mindist/dist)
                tempd += (mindist/dist)
            # 避免除以 0 的情况
            if tempd != 0:
                fluidD[fnodeID].vx = new_vx / tempd
                fluidD[fnodeID].vy = new_vy / tempd
                fluidD[fnodeID].vz = new_vz / tempd
            else:
                fluidD[fnodeID].vx = 0.0
                fluidD[fnodeID].vy = 0.0
                fluidD[fnodeID].vz = 0.0
            # print(f"AFTER*******Rank {rank}, 更新后流体点: {fluidD[fnodeID]}")
    
        local_ret.append((f"batch_{batch_id}_{i}_Fluid", fluidD))
    
    comm.Barrier() # 确保所有进程都完成了各自的局部计算之后，再进入下一步。只有当所有进程都“到达”这一点时，程序才会继续，这样可以保证数据的一致性。
    all_rets = comm.gather(local_ret, root=0)
    
    # if rank == 0:
    #     if all_rets:
    #         print(f"[DEBUG] Rank 0 收集到所有进程的计算结果，进程数: {len(all_rets)}")
    #         for proc_id, proc_data in enumerate(all_rets):
    #             print(f"  Process {proc_id} 贡献了 {len(proc_data)} 组流体数据")
    #     else:
    #         print("[DEBUG] 没有收集到结果!")
    
    if rank == 0:
        batch_rets = {}
        for process_ret in all_rets:
            for key, data in process_ret:
                batch_rets[key] = data
        print(f"[DEBUG] Batch {batch_id} 键值: {list(batch_rets.keys())[:10]}")
        print(f"[DEBUG] Batch {batch_id} 数据规模: {len(batch_rets)}")
        return batch_rets
    else:
        return None


def Compute_Velocity():
    """
    计算流体速度插值：
      1. 读取边界网格数据，并将数据划分为 k 个批次；
      2. 对每个批次调用 parallel_NNI_Velocity 进行插值计算，并等待所有进程完成；
      3. 在 root 进程，将当前批次结果写入磁盘，并更新索引；
      4. 批次间使用 Barrier 同步，确保前一批次完成后才进行下一批次的计算。
    """
    k = 8
    nnk = 2

    # 主进程处理数据加载
    if rank == 0:
        # 仅主进程加载数据
        BoundaryMeshLF = GetBoundDataF("/home/wangjh/miaox/FSI_Interpolation/jsondata/ringdict_10_F.json")
        BoundaryMeshL = GetBoundDataS("/home/wangjh/miaox/FSI_Interpolation/jsondata/ringdict_10_P.json", BoundaryMeshLF)
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


    index = {}  # 用于记录各批次结果文件路径
    for batch_id, batch in enumerate(batches):
        if rank == 0:
            print(f"\nProcessing batch {batch_id+1}/{len(batches)}，当前批次 [FluidD, SolidD] 对象数量: {len(batch)}")
        comm.Barrier()  # 同步：确保所有进程在开始新的批次前处于同一状态
        batch_rets = parallel_NNI_Velocity(batch, comm, rank, size, batch_id, nnk)
        
        if rank == 0:
            if batch_rets:
                print(f"[DEBUG] Batch {batch_id} 包含 {len(batch_rets)} 个结果.")
            else:
                print(f"[DEBUG] Batch {batch_id} 结果为空!")
            # 保存当前批次的结果到磁盘
            file_path = save_batch_to_disk(batch_rets, batch_id)
            index[batch_id] = file_path
        
        comm.Barrier()  # 确保各进程在开始下个批次前均已完成当前批次

    if rank == 0:
        print(f"\n[DEBUG] Forest index 包含 {len(index)} 个条目.")
        for bid, file_path in index.items():
            print(f"  Batch {bid} 存储在: {file_path}")
        save_forest_index(index)
        print(f"Total batches saved: {len(index)}")
        return index
    else:
        return None



def NNI_test_compute_velocity():
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
    # Compute_Velocity()
    NNI_test_compute_velocity()
