import numpy as np
import os
import shutil

def read_matrix(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    label = lines[0].strip()
    mat = np.array([[int(x) for x in line.strip().split()] for line in lines[1:]])
    return label, mat

def write_matrix(file_path, label, mat):
    with open(file_path, "w") as f:
        f.write(f"{label}\n")
        for row in mat:
            f.write(" ".join(str(x) for x in row) + "\n")

def shift_matrix(mat, dx, dy):
    res = np.zeros_like(mat)
    h, w = mat.shape

    # 计算重叠区域的起止坐标
    if dx >= 0:
        src_x1, dst_x1 = 0, dx
        src_x2, dst_x2 = w - dx, w
    else:
        src_x1, dst_x1 = -dx, 0
        src_x2, dst_x2 = w, w + dx

    if dy >= 0:
        src_y1, dst_y1 = 0, dy
        src_y2, dst_y2 = h - dy, h
    else:
        src_y1, dst_y1 = -dy, 0
        src_y2, dst_y2 = h, h + dy

    # 复制有效区域
    res[dst_y1:dst_y2, dst_x1:dst_x2] = mat[src_y1:src_y2, src_x1:src_x2]
    return res

n = int(input("输入n："))
data_dir = "data"
# 找到当前最大编号
existing_files = [int(name) for name in os.listdir(data_dir) if name.isdigit()]
max_cnt = max(existing_files) if existing_files else 0
new_cnt = max_cnt + 1

# 上下左右平移1格和2格
shifts = [(-1,0),(1,0),(0,-1),(0,1),(-2,0),(2,0),(0,-2),(0,2)]

for i in range(1, n+1):
    file_path = os.path.join(data_dir, str(i))
    label, mat = read_matrix(file_path)
    for dx, dy in shifts:
        shifted = shift_matrix(mat, dx, dy)
        write_matrix(os.path.join(data_dir, str(new_cnt)), label, shifted)
        new_cnt += 1