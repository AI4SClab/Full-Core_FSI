from mpi4py import MPI
import tracemalloc, time, os, sys, json, heapq

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


def update_fulid_velocity(nodes, nodef, M, R):
    """处理相同层的中间节点"""
    """使用堆优化最近邻查找，处理相同层的叶子节点"""
    for i in range(M):
        # 获取当前固体和流体节点的网格点
        solid_points = nodes.children[i].points
        fluid_points = nodef.children[i].points

        # 遍历每个固体点，使用堆查找最近的流体点
        for fluid_point in fluid_points:
            # 构建最小堆，用于动态维护最近的流体点
            min_heap = []
            for solid_point in solid_points:
                # 计算固体点与流体点之间的距离平方
                dist_squared = (
                    (solid_point.x - fluid_point.x) ** 2
                    + (solid_point.y - fluid_point.y) ** 2
                    + (solid_point.z - fluid_point.z) ** 2
                )
                # 使用堆存储距离和流体点压力值
                heapq.heappush(
                    min_heap,
                    (dist_squared, solid_point.vx, solid_point.vz, solid_point.vz),
                )

            # 从堆中取出距离最小的点
            nearest_dist, nearest_vx, nearest_vy, nearest_vz = heapq.heappop(min_heap)
            fluid_point.vx = nearest_vx  # 更新固体点的压力值
            fluid_point.vz = nearest_vy
            fluid_point.vz = nearest_vz

    return nodef  # 返回更新后的流体非叶子结点


def traverse_send_layers_StoF(FulidTree, SolidTree, N, M, R):
    """遍历两棵树，获取匹配的流体和固体树节点并进行插值计算，为流体网格点计算速度值"""
    newNodesL = []
    for layer_index in range(N):
        solid_node = SolidTree.children[layer_index]
        fluid_node = FulidTree.children[layer_index]

        new_fulid_node = update_fulid_velocity(solid_node, fluid_node, M, R)
        newNodesL.append(new_fulid_node)  # 收集结果

    return newNodesL


def interpolate_solid_to_fluid(num_batches, N, M, z_min, z_max, R):  # , output_dir):
    """并行加载和处理森林&为流体网格点计算速度值的的主逻辑"""

    forest_index = {}  # 森林索引
    for batch_id in range(num_batches):
        if rank == 0:
            # 主进程加载一个批次的森林
            batch_forest = load_batch_from_disk(batch_id)  # , output_dir)
            forest_keys = list(batch_forest.keys())
            num_trees = len(forest_keys)
            print(f"Batch {batch_id} loaded, contains {num_trees} trees.")
            # 将任务分成每两个树为一组
            tasks = [
                (forest_keys[group_id * 2], forest_keys[group_id * 2 + 1])
                for group_id in range(num_trees // 2)
            ]
            # print(f"tasks = {tasks}")
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

                # 以流体树的叶子节点为单位计算流体速度值
                newNodesL = traverse_send_layers_StoF(fluid_tree, solid_tree, N, M, R)
                # 更新树结构
                update_Tree(fluid_tree, newNodesL, z_min, z_max, N)
                # print_rtree(fluid_tree)
                # 保存结果
                results.append((fluid_tree_key, fluid_tree))
            elif rank == 0:
                # 主进程发送其他进程所需的树
                comm.send(
                    batch_forest[fluid_tree_key], dest=group_id % size, tag=group_id * 2
                )
                comm.send(
                    batch_forest[solid_tree_key],
                    dest=group_id % size,
                    tag=group_id * 2 + 1,
                )

        # 将所有进程的结果发送到主进程
        all_results = comm.gather(results, root=0)
        # 主进程更新 batch_forest
        if rank == 0:
            for result_list in all_results:
                for fluid_tree_key, fluid_tree in result_list:
                    batch_forest[fluid_tree_key] = fluid_tree

            # 保存当前批次的森林到磁盘
            print(f"rank{rank}...执行save_batch_to_disk函数...")
            file_path = save_batch_to_disk(batch_forest, batch_id)
            # 更新索引
            forest_index[batch_id] = file_path

        # 同步所有进程
        comm.Barrier()
        if rank == 0:
            print(f"Batch {batch_id} completed.")

    if rank == 0:
        # 调试：打印 index 的内容
        print(f"[DEBUG] Forest index contains {len(forest_index)} entries.")
        for batch_id, file_path in forest_index.items():
            print(f"  Batch {batch_id} stored at: {file_path}")

    if rank == 0:
        # 保存森林索引到磁盘
        save_forest_index(forest_index)
        print(f"Total batches saved: {len(forest_index)}")
        return forest_index
    else:
        return None


def fore_compute_Velocity():
    # 仅在 Rank 0 汇总时间和内存数据作为示例
    if rank == 0:
        print("Starting Compute_Pressure test...")

    tracemalloc.start()  # 开始内存追踪
    start_time = time.time()

    # 调用并行计算函数，注意各进程都需要调用 Compute_Velocity
    interpolate_solid_to_fluid(
        num_batches=8,  # , output_dir="/work1/wangjue/miaoxue/FSI_Interpolation/forest_batches"
        N=5,
        M=6,
        z_min=0,
        z_max=10,
        R=0.5,
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


if __name__ == "__main__":
    fore_compute_Velocity()
