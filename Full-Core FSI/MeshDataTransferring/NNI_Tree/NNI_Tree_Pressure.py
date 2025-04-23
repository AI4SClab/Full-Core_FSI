from mpi4py import MPI
import tracemalloc,time, os, sys, json,heapq

# 获取父目录路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将父目录添加到 sys.path
sys.path.append(parent_dir)

from TreeClass import TreeNode, update_Tree, print_rtree
from saveandloadFuns import load_batch_from_disk, save_batch_to_disk, save_forest_index


# 设置 MPI 环境
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# def update_solid_pressures(nodes, nodef, M, R):
#     """使用堆优化最近邻查找，处理相同层的叶子节点"""
#     for i in range(M):
#         # 获取当前固体和流体节点的网格点
#         solid_points = nodes.children[i].points
#         fluid_points = nodef.children[i].points

#         # 遍历每个固体点，使用堆查找最近的流体点
#         for solid_point in solid_points:
#             # 构建最小堆，用于动态维护最近的流体点
#             min_heap = []

#             for fluid_point in fluid_points:
#                 # 计算固体点与流体点之间的距离平方
#                 dist_squared = (
#                     (solid_point.x - fluid_point.x) ** 2
#                     + (solid_point.y - fluid_point.y) ** 2
#                     + (solid_point.z - fluid_point.z) ** 2
#                 )
#                 # 使用堆存储距离和流体点压力值
#                 heapq.heappush(min_heap, (dist_squared, fluid_point.pressure))

#             # 从堆中取出距离最小的点
#             nearest_dist, nearest_pressure = heapq.heappop(min_heap)
#             solid_point.pressure = nearest_pressure  # 更新固体点的压力值

#     return nodes  # 返回更新后的固体非叶子结点
def update_solid_pressures(nodes, nodef, M, R):
    """
    使用堆优化最近邻查找，处理相同层的叶子节点，并基于两个最近邻的距离加权更新固体点的压力值。
    
    参数:
        nodes: 固体节点树
        nodef: 流体节点树
        M: 固体和流体节点的数量
        R: 加权距离阈值 (未使用，但保留用于扩展)
    
    返回:
        更新后的固体节点树
    """
    for i in range(M):
        # 获取当前固体和流体节点的网格点
        solid_points = nodes.children[i].points
        fluid_points = nodef.children[i].points

        # 遍历每个固体点，使用堆查找最近的两个流体点
        for solid_point in solid_points:
            # 构建最小堆，用于动态维护最近的流体点
            min_heap = []

            for fluid_point in fluid_points:
                # 计算固体点与流体点之间的距离平方
                dist_squared = (
                    (solid_point.x - fluid_point.x) ** 2
                    + (solid_point.y - fluid_point.y) ** 2
                    + (solid_point.z - fluid_point.z) ** 2
                )
                # 使用堆存储 (距离平方, 流体点压力值)
                heapq.heappush(min_heap, (dist_squared, fluid_point.pressure))

            # 从堆中取出距离最近的两个点
            nearest_neighbors = heapq.nsmallest(2, min_heap)

            if len(nearest_neighbors) == 2:
                # 提取最近邻的距离和压力值
                (dist1, pressure1), (dist2, pressure2) = nearest_neighbors

                # 防止除以零，确保距离不为零
                if dist1 == 0:
                    weight1, weight2 = 1, 0
                elif dist2 == 0:
                    weight1, weight2 = 0, 1
                else:
                    # 距离的加权权重（使用距离的倒数作为权重）
                    weight1 = 1 / dist1
                    weight2 = 1 / dist2
                    weight_sum = weight1 + weight2
                    weight1 /= weight_sum
                    weight2 /= weight_sum

                # 根据加权更新固体点的压力值
                solid_point.pressure = weight1 * pressure1 + weight2 * pressure2

            elif len(nearest_neighbors) == 1:
                # 如果只有一个最近邻，直接使用其压力值
                dist1, pressure1 = nearest_neighbors[0]
                solid_point.pressure = pressure1

    return nodes  # 返回更新后的固体节点树


def traverse_send_layers_FtoS(SolidTree, FulidTree, N, M, R):
    """遍历两棵树，获取匹配的流体和固体树节点并进行插值计算，为固体网格点计算压力值"""
    newNodesL = []
    for layer_index in range(N):
        solid_node = SolidTree.children[layer_index]
        fluid_node = FulidTree.children[layer_index]
        # print(f"......固体和流体树节点是否匹配：[{solid_node.z_min},{solid_node.z_max}], [{fluid_node.z_min},{fluid_node.z_max}]")
        new_solid_node = update_solid_pressures(
            solid_node, fluid_node, M, R
        )  # 得到更新后的非叶子节点
        newNodesL.append(new_solid_node)
    return newNodesL  # 固体树结构的所有非叶子节点，即树的第1层节点，根节点为第0层



def interpolate_fluid_to_solid(num_batches, N,M, z_min, z_max, R):#, output_dir): 
    """并行加载和处理森林&为固体网格点计算压力值的的主逻辑"""

    forest_index = {}  # 森林索引
    for batch_id in range(num_batches): 
        if rank == 0:
            # 主进程加载一个批次的森林
            batch_forest = load_batch_from_disk(batch_id)#, output_dir)
            forest_keys = list(batch_forest.keys())
            num_trees = len(forest_keys)
            print(f"Batch {batch_id} loaded, contains {num_trees} trees.")
            # 将任务分成每两个树为一组
            tasks = [
                (forest_keys[group_id * 2], forest_keys[group_id * 2 + 1])
                for group_id in range(num_trees // 2)
            ]
        else:
            batch_forest = None
            tasks = None

        # 广播任务列表给所有进程
        tasks = comm.bcast(tasks if rank == 0 else None, root=0)
        
        results = []  # 保存本地进程的计算结果
        # 分发任务并计算
        for group_id, (fluid_tree_key, solid_tree_key) in enumerate(tasks):
            if group_id % size == rank:
                if rank == 0:
                    # 主进程直接从本地获取树
                    fluid_tree = batch_forest[fluid_tree_key]
                    solid_tree = batch_forest[solid_tree_key]
                else:
                    # 从主进程接收树
                    # print(f"rank{rank}, {fluid_tree_key}, {solid_tree_key}, {batch_forest}")
                    fluid_tree = comm.recv(source=0, tag=group_id * 2)
                    solid_tree = comm.recv(source=0, tag=group_id * 2 + 1)

                # 以固体树的叶子节点为单位计算固体压力值
                newNodesL = traverse_send_layers_FtoS(solid_tree, fluid_tree, N, M, R)
                # 更新树结构
                update_Tree(solid_tree, newNodesL, z_min, z_max, N)
                # print_rtree(solid_tree)
                # 保存结果
                results.append((solid_tree_key, solid_tree))
            elif rank == 0:
                # 主进程发送其他进程所需的树
                comm.send(batch_forest[fluid_tree_key], dest=group_id % size, tag=group_id * 2)
                comm.send(batch_forest[solid_tree_key], dest=group_id % size, tag=group_id * 2 + 1)
        
        # 将所有进程的结果发送到主进程
        all_results = comm.gather(results, root=0)
        # 主进程更新 batch_forest
        if rank == 0:
            for result_list in all_results:
                for solid_tree_key, solid_tree in result_list:
                    batch_forest[solid_tree_key] = solid_tree

            # 保存当前批次的森林到磁盘
            print(f"rank{rank}...执行save_batch_to_disk函数...")
            file_path = save_batch_to_disk(batch_forest, batch_id)
            # 更新索引
            forest_index[batch_id] = file_path

        # 同步所有进程      
        comm.Barrier()
    #     if rank == 0:
    #         print(f"Batch {batch_id} completed.")
    
    # if rank == 0:
    #     # 调试：打印 index 的内容
    #     print(f"[DEBUG] Forest index contains {len(forest_index)} entries.")
    #     for batch_id, file_path in forest_index.items():
    #         print(f"  Batch {batch_id} stored at: {file_path}")

    # if rank == 0:
    #     # 保存森林索引到磁盘
    #     save_forest_index(forest_index)
    #     print(f"Total batches saved: {len(forest_index)}")
    #     return forest_index
    # else:
    #     return None



def fore_compute_Pressure():
    # 仅在 Rank 0 汇总时间和内存数据作为示例
    if rank == 0:
        print("Starting Compute_Pressure test...")

    tracemalloc.start()  # 开始内存追踪
    start_time = time.time()
    
    # 调用并行计算函数，注意各进程都需要调用 Compute_Velocity
    interpolate_fluid_to_solid(
        num_batches=1, #, output_dir="/work1/wangjue/miaoxue/FSI_Interpolation/forest_batches"
        N =135,
        M = 6,
        z_min = 0,
        z_max = 1350,
        R = 0.5,
    )
    
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
    if rank != 0:
        # 非 Rank 0 的进程将标准输出重定向到 `/dev/null`
        sys.stdout = open(os.devnull, "w")


if __name__ == "__main__":
    fore_compute_Pressure()
