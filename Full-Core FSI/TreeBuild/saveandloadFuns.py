# -*- coding: utf-8 -*-
import os
import pickle, dill
from TreeClass import TreeNode, MeshPoint, print_rtree


def save_batch_to_disk(
    batch_forest,
    batch_id,
    output_dir="D:/Code/FSI_Interpolation2/forest_batches",
):
    """将一个批次的森林存储到磁盘"""
    os.makedirs(output_dir, exist_ok=True)  # 创建存储目录
    file_path = os.path.join(output_dir, f"forest_batch_{batch_id}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(batch_forest, f)
    print(f"Batch {batch_id} saved to {file_path}.")
    return file_path


def load_batch_from_disk(batch_id, output_dir="D:/Code/FSI_Interpolation2/forest_batches"):
    """从磁盘加载一个批次的森林"""
    file_path = os.path.join(output_dir, f"forest_batch_{batch_id}.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Batch file not found: {file_path}")

    with open(file_path, "rb") as f:
        # batch_forest = pickle.load(f)
        batch_forest = dill.load(f)  # dill 对复杂对象的支持更好
    print(f"Batch {batch_id} loaded from {file_path}.")
    return batch_forest


def load_tree_from_forest(
    tree_key, output_dir="D:/Code/FSI_Interpolation2/forest_batches"
):
    """动态加载特定树"""
    # 提取批次编号
    batch_id = int(
        tree_key.split("_")[1]
    )  # 假设键格式为 batch_{batch_id}_tree_{tree_id}_{type}
    batch_forest = load_batch_from_disk(batch_id, output_dir)
    # 返回目标树
    return batch_forest[tree_key]


def save_forest_index(
    index, output_dir="D:/Code/FSI_Interpolation2/forest_batches"
):
    """保存森林索引到磁盘"""
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(
        output_dir,
        "D:/Code/FSI_Interpolation2/forest_batches/forest_index.pkl",
    )
    with open(index_path, "wb") as f:
        pickle.dump(index, f)
    print(f"Forest index saved to {index_path}.")
    return index_path


def load_forest_index(output_dir="D:/Code/FSI_Interpolation2/forest_batches"):
    """加载森林索引"""
    index_path = os.path.join(output_dir, "forest_index.pkl")
    with open(index_path, "rb") as f:
        # forest_index = pickle.load(f)
        forest_index = dill.load(f)
    print(f"Forest index loaded from {index_path}.")
    return forest_index


if __name__ == "__main__":
    output_dir = "D:/Code/FSI_Interpolation2/forest_batches"
    # forest_index = load_forest_index(output_dir)
    # batch_size = len(forest_index)
    # print(forest_index)
    count = 0
    num = 0
    for i in range(8):
        batch_forest = load_batch_from_disk(i)
        # print(len(batch_forest),batch_forest)
        for key, value in batch_forest.items():
            print(key, len(value))
            num += len(value)
            # break
        print(len(batch_forest))
        count += len(batch_forest)
    print(f"树总数：{count}, 网格点总数：{num}")
    # print(f"打印流体树.................................")
    # print_rtree(batch_forest["batch_0_tree_0_fluid"])
    # print(f"打印固体树.................................")
    # print_rtree(batch_forest["batch_0_tree_0_solid"])
    # for key, value in batch_forest.items():
    #     print(key, len(value))