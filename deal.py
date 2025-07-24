# augment_data.py
import os
import numpy as np
from PIL import Image

# -------------------- 参数 --------------------
GRID_SIZE = 10          # 10×10 二值图
OPS = {
    'shift_up_1':    (0, -1),
    'shift_up_2':    (0, -2),
    'shift_down_1':  (0,  1),
    'shift_down_2':  (0,  2),
    'shift_left_1':  (-1, 0),
    'shift_left_2':  (-2, 0),
    'shift_right_1': (1,  0),
    'shift_right_2': (2,  0),
    'rot_1':  1,
    'rot_-1': -1,
    'rot_2':  2,
    'rot_-2': -2,
}

# -------------------- 工具函数 --------------------
def load_matrix(path):
    """读取文件 -> (label, 10×10 np.uint8 array)"""
    with open(path, encoding='utf-8') as f:
        label = int(f.readline().strip())
        mat = np.array([[int(ch) for ch in line.strip()] for line in f])
    return label, mat

def save_matrix(path, label, mat):
    """把 (label, 10×10 array) 写回文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{label}\n")
        for row in mat:
            f.write("".join(map(str, row)) + "\n")

def next_available_id():
    """返回 data 目录下一个可用整数文件名"""
    os.makedirs("data", exist_ok=True)
    files = [int(name) for name in os.listdir("data") if name.isdigit()]
    return max(files, default=0) + 1

# -------------------- 几何变换 --------------------
def shift(mat, dx, dy):
    """平移 dx, dy，越界填 0"""
    res = np.zeros_like(mat)
    src_h, src_w = mat.shape
    dst_y = max(0, -dy)
    dst_x = max(0, -dx)
    src_y = max(0,  dy)
    src_x = max(0,  dx)
    h = min(src_h - src_y, src_h - dst_y)
    w = min(src_w - src_x, src_w - dst_x)
    if h > 0 and w > 0:
        res[dst_y:dst_y+h, dst_x:dst_x+w] = mat[src_y:src_y+h, src_x:src_x+w]
    return res

def rotate(mat, angle):
    """旋转 angle 度，最近邻插值，空白填 0"""
    img = Image.fromarray((mat * 255).astype(np.uint8), mode='L')
    img = img.rotate(angle, fillcolor=0)
    return (np.array(img) > 127).astype(np.uint8)

# -------------------- 主逻辑 --------------------
def main():
    n = int(input("请输入要处理的文件个数 n："))

    start_id = next_available_id()
    current_id = start_id

    for idx in range(1, n + 1):
        src_path = os.path.join("data", str(idx))
        if not os.path.isfile(src_path):
            print(f"文件 {src_path} 不存在，跳过")
            continue
        label, mat = load_matrix(src_path)

        for op_name, param in OPS.items():
            if isinstance(param, tuple):
                new_mat = shift(mat, *param)
            else:
                new_mat = rotate(mat, param)

            save_matrix(os.path.join("data", str(current_id)), label, new_mat)
            print(f"已生成 {current_id}  <-  {idx}  {op_name}")
            current_id += 1

    print("全部完成！共生成", current_id - start_id, "个新文件")

if __name__ == "__main__":
    main()