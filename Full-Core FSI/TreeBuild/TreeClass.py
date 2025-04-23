import numpy as np


class MeshPoint:
    def __init__(
        self, ID=0, x=0.0, y=0.0, z=0.0, vx=None, vy=None, vz=None, pressure=None, neighborfaces=None
    ):
        """
        初始化一个三维点。

        参数:
        x (float): 点的 x 坐标。
        y (float): 点的 y 坐标。
        z (float): 点的 z 坐标。
        vx (float): 点在 x 方向的速度分量。
        vy (float): 点在 y 方向的速度分量。
        vz (float): 点在 z 方向的速度分量。
        pressure (float): 点的压力属性。
        """
        self.ID = ID
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.pressure = pressure
        self.neighborfaces = neighborfaces

    def __repr__(self):
        """
        返回点的字符串表示。
        """
        return (
            "Point(ID={ID}, x={x}, y={y}, z={z}, "
            "vx={vx}, vy={vy}, vz={vz}, "
            "pressure={pressure})".format(
                ID=self.ID,
                x=self.x,
                y=self.y,
                z=self.z,
                vx=self.vx,
                vy=self.vy,
                vz=self.vz,
                pressure=self.pressure,
            )
        )

    def move(self, dt):
        """
        根据速度分量移动点的位置。

        参数:
        dt (float): 时间增量，用于计算移动的距离。
        """
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

    def set_pressure(self, pressure):
        """
        设置点的压力属性。

        参数:
        pressure (float): 要设置的压力值。
        """
        self.pressure = pressure

    def get_pressure(self):
        """
        获取点的压力属性。

        返回:
        float: 点的压力值。
        """
        return self.pressure

    def __lt__(self, other):
        """
        定义小于号的比较规则，用于堆排序。

        参数:
        other (MeshPoint): 另一个 MeshPoint 实例。

        返回:
        bool: 如果当前点小于另一个点，则返回 True。
        """
        # 默认使用 ID 作为比较依据（可以根据需求更改为其他属性）
        return self.ID < other.ID


def compute_angle(x, y):
    """计算点相对于圆心 (0, 0) 的角度（以度为单位）"""
    return np.degrees(np.arctan2(y, x)) % 360


class TreeNode:
    """树节点类"""
    def __init__(self, z_min, z_max, angle_min, angle_max):
        self.z_min = z_min
        self.z_max = z_max
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.points = []  # 叶子节点保存数据点
        self.children = []  # 中间节点保存子节点

    def insert(self, point):
        """插入点到当前节点"""
        if self.is_leaf():
            self.points.append(point)
        else:
            # 插入到子节点
            for child in self.children:
                if child.contains(point):
                    child.insert(point)
                    return

            # 如果没有合适的子节点，创建一个新的子节点
            new_child = TreeNode(self.z_min, self.z_max, self.angle_min, self.angle_max)
            new_child.insert(point)
            self.children.append(new_child)

    def is_leaf(self):
        """检查当前节点是否为叶子节点"""
        return len(self.children) == 0

    def contains(self, point):
        """检查点是否在当前节点的范围内"""
        return (
            self.z_min <= point.z <= self.z_max
            and self.angle_min <= compute_angle(point.x, point.y) <= self.angle_max
        )


# def index_cylinder_points(points, z_min, z_max, N, M):
#     """构建 索引树结构"""
#     root = TreeNode(z_min, z_max, 0, 360)  # 根节点覆盖整个空间

#     # 计算每层的高度范围
#     layer_height = (z_max - z_min) / N
#     angle_step = 360 / M

#     # 创建中间节点和叶子节点
#     for i in range(N):
#         # 创建中间节点
#         mid_node = TreeNode(
#             z_min + i * layer_height, z_min + (i + 1) * layer_height, 0, 360
#         )

#         for j in range(M):
#             # 创建叶子节点
#             angle_range_min = j * angle_step
#             angle_range_max = angle_range_min + angle_step
#             leaf_node = TreeNode(
#                 z_min + i * layer_height,
#                 z_min + (i + 1) * layer_height,
#                 angle_range_min,
#                 angle_range_max,
#             )

#             # 将叶子节点插入到中间节点
#             mid_node.children.append(leaf_node)

#         # 将中间节点插入到根节点
#         root.children.append(mid_node)

#     # 将点插入到相应的叶子节点
#     for point in points:
#         layer_index = int((point.z - z_min) / layer_height)
#         if layer_index >= N:
#             layer_index = N - 1

#         angle_index = int(compute_angle(point.x, point.y) / angle_step)
#         leaf_node = root.children[layer_index].children[angle_index]
#         leaf_node.insert(point)

#     return root


def index_cylinder_points(points, z_min, z_max, N, M):
    """构建索引树结构"""
    root = TreeNode(z_min, z_max, 0, 360)  # 根节点覆盖整个空间

    # 计算每层的高度范围和角度步长
    layer_height = (z_max - z_min) / N
    angle_step = 360 / M

    # 创建中间节点和叶子节点
    for i in range(N):
        # 创建中间节点
        mid_node = TreeNode(
            z_min + i * layer_height, z_min + (i + 1) * layer_height, 0, 360
        )
        mid_node.children = []  # 初始化中间节点的 children 列表

        for j in range(M):
            # 创建叶子节点
            angle_range_min = j * angle_step
            angle_range_max = angle_range_min + angle_step
            leaf_node = TreeNode(
                z_min + i * layer_height,
                z_min + (i + 1) * layer_height,
                angle_range_min,
                angle_range_max,
            )
            mid_node.children.append(leaf_node)  # 添加叶子节点

        root.children.append(mid_node)  # 添加中间节点

    # 将点插入到相应的叶子节点
    for point in points:
        # 计算 layer_index
        layer_index = int((point.z - z_min) / layer_height)
        if layer_index < 0:
            layer_index = 0
        elif layer_index >= N:
            layer_index = N - 1

        # 计算 angle_index
        angle = compute_angle(point.x, point.y)
        angle_index = int(angle / angle_step)
        if angle_index >= M:
            angle_index = M - 1

        # 插入点到叶子节点
        leaf_node = root.children[layer_index].children[angle_index]
        leaf_node.insert(point)

    return root


def print_rtree(node, depth=0):
    """递归打印 树节点并统计网格点总数"""
    # global total_points  # 使用全局变量来统计总点数
    # total_points += len(node.points)  # 累加当前节点的点数

    # print("  " * depth + f"Depth: {depth} Node: Z({node.z_min}, {node.z_max}), Angle({node.angle_min}, {node.angle_max}), Points: {len(node.points)}")
    # print(f"{'  ' * depth}len(node.children): {len(node.children)}")
    print(
        "  " * depth
        + "Depth: {depth} Node: Z({z_min}, {z_max}), Angle({angle_min}, {angle_max}), Points: {points}".format(
            depth=depth,
            z_min=node.z_min,
            z_max=node.z_max,
            angle_min=node.angle_min,
            angle_max=node.angle_max,
            points=len(node.points),
        )
    )
    print("{}len(node.children): {}".format("  " * depth, len(node.children)))

    # 如果是叶子节点，打印所有的点
    if node.is_leaf():
        for point in node.points:
            # print(f"{'  ' * (depth + 1)}Point: ({point.x}, {point.y}, {point.z})")
            # print(f"{'  ' * (depth + 1)}point: ({point})")
            print("{}point: ({})".format("  " * (depth + 1), point))

    for child in node.children:
        print_rtree(child, depth + 1)


def update_Tree(Tree, newNodesL, z_min, z_max, N):
    # 根据得到的更新后的树结点更新内存中的树结构
    layer_height = (z_max - z_min) / N

    for newnode in newNodesL:
        layer_index = int((newnode.z_min - z_min) / layer_height)
        try:
            if newnode.z_min == Tree.children[layer_index].z_min:
                Tree.children[layer_index] = newnode
        except Exception as e:
            print("Error retrieving result: ", e)


