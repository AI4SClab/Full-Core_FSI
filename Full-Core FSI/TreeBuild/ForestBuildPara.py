from mpi4py import MPI
import random, json, sys, os, tracemalloc, time
from typing import Dict

# 获取父目录路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将父目录添加到 sys.path
sys.path.append(parent_dir)

from saveandloadFuns import save_forest_index, save_batch_to_disk
from TreeClass import MeshPoint, index_cylinder_points, print_rtree
from AuxGeomComp import split_into_batches, GetBoundFaceData


# 设置 MPI 环境
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def GetBoundDataF(boundface_f_dict, jsonfileF):
    # 定义最小和最大值
    min_pre = -56.54605
    max_pre = 42.18494

    BoundaryMeshL = []
    # f_dic: Dict[int, MeshPoint] = {}
    # 从文件中读取字典
    with open(jsonfileF, "r", encoding="utf-8") as f:
        ring_dict = json.load(f)
        print(f"rank{rank}:loadedL中的字典数{len(ring_dict)}....")
        
        for center, boudL in ring_dict.items():
            FulidL = []
            for [ID, X,Y,Z,] in boudL:
                neighborfaces = boundface_f_dict[str(ID)]
                pressure = random.uniform(min_pre, max_pre)
                node = MeshPoint(ID, X,Y,Z, pressure = pressure, neighborfaces = neighborfaces)
                # f_dic[ID]=node
                FulidL.append(node)
            
            BoundaryMeshL.append([FulidL])
    return BoundaryMeshL


def GetBoundDataS(boundface_s_dict, jsonfileS,BoundaryMeshLF):
    # 定义最小和最大值
    minv = -0.05
    maxv = 0.05
    # s_dic: Dict[int, MeshPoint] = {}
    
    # 从文件中读取字典
    with open(jsonfileS, "r", encoding="utf-8") as f:
        ring_dict = json.load(f)
        print(f"rank{rank}:loadedL中的字典数{len(ring_dict)}....")
        # for ring_dict in loadedL:
        i = 0
        for center, boudL in ring_dict.items():
            SolidL = []
            
            for [ID, X,Y,Z,] in boudL:
                neighborfaces = boundface_s_dict[str(ID)]
                vx = random.uniform(minv, maxv)
                vy = random.uniform(minv, maxv)
                vz = random.uniform(minv, maxv)
                node = MeshPoint(ID, X,Y,Z, vx = vx, vy=vy,vz=vz, neighborfaces=neighborfaces)
                SolidL.append(node)
                
            BoundaryMeshLF[i].append(SolidL)
            i += 1
            
    return BoundaryMeshLF



def parallel_index_cylinder_points(batch, z_min, z_max, N, M, comm, rank, size, batch_id):
    """并行处理两个数据集并构建树索引结构"""
    # S = len(batch)
    # if size < S * 2:
    #     raise ValueError("Not enough processes for this batch. Each element requires 2 processes.")
    
    local_trees = []
    for i, [fluidL, solidL] in enumerate(batch):
        fluid_process = i * 2
        solid_process = i * 2 + 1
        
        if rank == fluid_process:
            fluidTree = index_cylinder_points(fluidL, z_min, z_max, N, M)
            # print(f"*******Rank {rank}:打印流体树：")
            # print_rtree(fluidTree)
            print(f"*******Rank {rank}: 流体索引树构建完毕")
            local_trees.append((f"batch_{batch_id}_tree_{i}_fluid", fluidTree))#{"type": "fluid", "tree": fluidTree}))
        
        elif rank == solid_process:
            solidTree = index_cylinder_points(solidL, z_min, z_max, N, M)
            # print(f"*******Rank {rank}:打印固体树：")
            # print_rtree(solidTree)
            print(f".....Rank {rank}: 固体索引树构建完毕")
            local_trees.append((f"batch_{batch_id}_tree_{i}_solid", solidTree))# {"type": "solid", "tree": solidTree}))
    
    comm.Barrier()
    all_trees = comm.gather(local_trees, root=0)

    # 调试：打印 all_trees 的结构信息
    if rank == 0:
        if all_trees:
            print(f"[DEBUG] All trees gathered in rank 0: {len(all_trees)} processes contributed.")
            for i, process_trees in enumerate(all_trees):
                print(f"  Process {i} contributed {len(process_trees)} trees.")
                if process_trees:  # 如果当前进程有树数据
                    print(f"    Example tree key: {process_trees[0][0]}")
        else:
            print("[DEBUG] All trees is empty!")

    if rank == 0:
        batch_forest = {}
        for process_trees in all_trees:
            for key, tree_data in process_trees:
                batch_forest[key] = tree_data

        # 调试：打印 batch_forest的结构信息
        print(f"[DEBUG] Batch {batch_id} forest keys: {list(batch_forest.keys())[:10]}")  # 打印前 5 个键
        print(f"[DEBUG] Batch {batch_id} forest size: {len(batch_forest)}")
        return batch_forest
    else:
        return None


def forest_build():
    N = 135
    M = 6
    z_min = 0
    z_max = 1350
    # R = 0.5
    k = 1
    # k = 4
    # k = 2
    # k = 1

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 主进程处理数据加载
    if rank == 0:
        boundface_s_dict = GetBoundFaceData("D:/Code/FSI_Interpolation2/jsondata/1.6W-12.6W/1facedict_1350_P_1.6W.json")
        boundface_f_dict = GetBoundFaceData("D:/Code/FSI_Interpolation2/jsondata/1.6W-12.6W/1facedict_1350_F_12.6W.json")
        # boundface_s_dict = GetBoundFaceData("D:/Code/FSI_Interpolation2/jsondata/4.8W-37.8W/1facedict_1350_P_4.8W.json")
        # boundface_f_dict = GetBoundFaceData("D:/Code/FSI_Interpolation2/jsondata/4.8W-37.8W/1facedict_1350_F_37.8W.json")
        # boundface_s_dict = GetBoundFaceData("D:/Code/FSI_Interpolation2/jsondata/16.8W-124W/1facedict_1350_P_16.8W.json")
        # boundface_f_dict = GetBoundFaceData("D:/Code/FSI_Interpolation2/jsondata/16.8W-124W/1facedict_1350_F_124W.json")
        # 仅主进程加载数据
        BoundaryMeshLF = GetBoundDataF(boundface_f_dict, "D:/Code/FSI_Interpolation2/jsondata/1.6W-12.6W/1ringdict_1350_F_12.6W.json")
        BoundaryMeshL = GetBoundDataS(boundface_s_dict, "D:/Code/FSI_Interpolation2/jsondata/1.6W-12.6W/1ringdict_1350_P_1.6W.json", BoundaryMeshLF)
        # BoundaryMeshLF = GetBoundDataF(boundface_f_dict, "D:/Code/FSI_Interpolation2/jsondata/4.8W-37.8W/1ringdict_1350_F_37.8W.json")
        # BoundaryMeshL = GetBoundDataS(boundface_s_dict, "D:/Code/FSI_Interpolation2/jsondata/4.8W-37.8W/1ringdict_1350_P_4.8W.json", BoundaryMeshLF)
        # BoundaryMeshLF = GetBoundDataF(boundface_f_dict, "D:/Code/FSI_Interpolation2/jsondata/16.8W-124W/1ringdict_1350_F_124W.json")
        # BoundaryMeshL = GetBoundDataS(boundface_s_dict, "D:/Code/FSI_Interpolation2/jsondata/16.8W-124W/1ringdict_1350_P_16.8W.json", BoundaryMeshLF)
        # print(f"rank{rank}:需要构建的流体树及固体树总数: {len(BoundaryMeshL)*2}")
        
        # 将 BoundaryMeshL 按批次划分
        batches = split_into_batches(BoundaryMeshL, k)

        print(f"Total processes: {size}")
        print(f"BoundaryMeshL: {len(BoundaryMeshL)}")
        print(f"Number of batches: {k}, len(batches): {len(batches)}")
        
    else:
        # 非主进程初始化为空
        batches = None

    # 广播 batches 到所有进程
    batches = comm.bcast(batches, root=0)

    # 确保所有进程同步 
    comm.Barrier()

    index = {}  # 森林索引
    for batch_id, batch in enumerate(batches):
        if rank == 0:
            print(f"Processing batch {batch_id + 1}/{len(batches)}, 当前批次中[FulidL, SolidL]元素数量：{len(batches[batch_id])}=={len(batch)}")
        
        # 所有进程同步
        comm.Barrier()

        # 并行构建批次森林
        batch_forest = parallel_index_cylinder_points(batch, z_min, z_max, N, M, comm, rank, size, batch_id)
        
        if rank == 0:
            # 调试：打印 batch_forest 的内容
            if batch_forest:
                print(f"[DEBUG] Batch {batch_id} forest contains {len(batch_forest)} trees.")
                example_key = list(batch_forest.keys())[0]
                print(f"  Example tree key: {example_key}, Tree type: {batch_forest[example_key]}")
            else:
                print(f"[DEBUG] Batch {batch_id} forest is empty!")

            # 保存当前批次的森林到磁盘
            file_path = save_batch_to_disk(batch_forest, batch_id)
            # 更新索引
            index[batch_id] = file_path

    # 主进程处理最终的森林索引保存
    if rank == 0:
        # 调试：打印 index 的内容
        print(f"[DEBUG] Forest index contains {len(index)} entries.")
        for batch_id, file_path in index.items():
            print(f"  Batch {batch_id} stored at: {file_path}")

        # 保存森林索引到磁盘
        save_forest_index(index)
        print(f"Total batches saved: {len(index)}")
        return index
    else:
        return None
    
    
    
def fore_compute_Pressure():
    # 仅在 Rank 0 汇总时间和内存数据作为示例
    if rank == 0:
        print("Starting Compute_Pressure test...")

    tracemalloc.start()  # 开始内存追踪
    start_time = time.time()
    
    # 调用并行计算函数，注意各进程都需要调用 Compute_Velocity
    forest_build()
    
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
    
if __name__ == "__main__":
    fore_compute_Pressure()
