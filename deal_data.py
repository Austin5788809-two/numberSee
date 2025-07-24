import numpy as np
import os
import random
from scipy.ndimage import rotate

matrix_size = 50

def read_data(file_path):
    with open(file_path, "r") as f:
        label = int(f.readline().strip())
        mat = []
        for _ in range(matrix_size):
            row = list(map(int, f.readline().strip().split()))
            mat.append(row)
        mat = np.array(mat, dtype=int)
    return label, mat

def write_data(file_path, label, mat):
    with open(file_path, "w") as f:
        f.write(f"{label}\n")
        for row in mat:
            f.write(" ".join(str(x) for x in row) + "\n")

def shift_matrix(mat, dx, dy):
    shifted = np.roll(mat, shift=dy, axis=0)
    shifted = np.roll(shifted, shift=dx, axis=1)
    # 清理越界部分
    if dy > 0:
        shifted[:dy, :] = 0
    elif dy < 0:
        shifted[dy:, :] = 0
    if dx > 0:
        shifted[:, :dx] = 0
    elif dx < 0:
        shifted[:, dx:] = 0
    return shifted

def add_noise(mat, num_points):
    mat = mat.copy()
    for _ in range(num_points):
        x = random.randint(0, matrix_size-1)
        y = random.randint(0, matrix_size-1)
        mat[x, y] = 1 if mat[x, y] == 0 else 0
    return mat

def augment(label, mat):
    augmented = []
    # 平移
    for dx in [-2, -1, 1, 2]:
        augmented.append(shift_matrix(mat, dx, 0))
    for dy in [-2, -1, 1, 2]:
        augmented.append(shift_matrix(mat, 0, dy))
    # 旋转
    for angle in [2, 3, 4, 5]:
        augmented.append(rotate(mat, angle, reshape=False, order=0, mode='constant', cval=0))
        augmented.append(rotate(mat, -angle, reshape=False, order=0, mode='constant', cval=0))
    # 添加噪点
    for n in [1, 2]:
        augmented.append(add_noise(mat, n))
    return [(label, m) for m in augmented]

def main(l, r):
    data_dir = "data"
    # 找到当前最大编号
    files = [f for f in os.listdir(data_dir) if f.isdigit()]
    max_id = max([int(f) for f in files]) if files else 0
    cur_id = max_id + 1
    print(f"start in {cur_id}")
    for idx in range(l, r+1):
        file_path = os.path.join(data_dir, str(idx))
        if not os.path.exists(file_path):
            continue
        label, mat = read_data(file_path)
        aug_data = augment(label, mat)
        for label, aug_mat in aug_data:
            write_data(os.path.join(data_dir, str(cur_id)), label, aug_mat)
            cur_id += 1

if __name__ == "__main__":
    l = int(input("输入l: "))
    r = int(input("输入r: "))
    main(l, r)