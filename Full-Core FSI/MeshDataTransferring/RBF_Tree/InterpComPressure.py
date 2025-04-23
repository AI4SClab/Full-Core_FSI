from mpi4py import MPI
import tracemalloc,time, sys, os

# 获取父目录路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将父目录添加到 sys.path
sys.path.append(parent_dir)

from RBFCompute import compute_pressure
from saveandloadFuns import save_batch_to_disk, save_forest_index, load_batch_from_disk
from TreeClass import TreeNode, update_Tree, print_rtree


# 设置 MPI 环境
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def update_solid_pressures(nodes, nodef, M, R):
    """处理相同层的叶子节点"""
    for i in range(M):
        # print(nodes.children[i].angle_min,nodef.children[i].angle_min)
        solid_points = nodes.children[i].points
        fluid_points = nodef.children[i].points
        # print(f"......固体和流体树节点是否匹配：[{nodes.children[i].angle_min},{nodes.children[i].angle_max}], [{nodef.children[i].angle_min},{nodef.children[i].angle_max}]")
        pressure_matrix_s = compute_pressure(
            solid_points, fluid_points, R
        )  # 为固体网格点计算压力值
        # print(f"len(solid_points), len(fluid_points), pressure_matrix_s.shape:{len(solid_points), len(fluid_points), pressure_matrix_s.shape}")
        # 更新固体结点的压力值
        # update_solid_pressures(nodes, i, pressure_matrix_s)
        # for j in range()
        for j in range(len(nodes.children[i].points)):
            nodes.children[i].points[j].pressure = pressure_matrix_s[
                j, 0
            ]  # 更新各个固体网格点的压力值

    return nodes  # 返回更新后的固体非叶子结点


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


def interpolate_fluid_to_solid(num_batches):#, output_dir): 
    """并行加载和处理森林&为固体网格点计算压力值的的主逻辑"""
    N = 135
    M = 6
    z_min = 0
    z_max = 1350
    R = 5
    num_batches = 1

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
            # print(f"tasks = {tasks}")
        else:
            batch_forest = None
            tasks = None

        # 广播任务列表给所有进程
        tasks = comm.bcast(tasks*10 if rank == 0 else None, root=0)
        
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
        num_batches=8 
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
    fore_compute_Pressure()