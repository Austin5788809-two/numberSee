import os
import numpy as np
import random
import sys
from PIL import Image

# 默认数据目录
data = sys.argv[1] if len(sys.argv) > 1 else "data"

def process_sample(filename: int):
    """处理单个样本，返回新文件名"""
    path = os.path.join(data, str(filename))
    if not os.path.exists(path):
        print(f"missing {path}")
        return None

    with open(path) as f:
        lines = [ln.rstrip() for ln in f]

    if len(lines) < 31:
        print("format error")
        return None

    one_hot = lines[0]
    matrix = np.array([[int(ch) for ch in ln if ch in '01'] for ln in lines[1:31]],
                      dtype=np.uint8)

    # ---- 随机选两种变换 ----
    transforms = ['shift', 'rotate', 'add_noise', 'flip']
    for t in random.sample(transforms, 2):
        if t == 'shift':
            d, amt = random.randint(1, 4), random.randint(1, 4)
            if d == 1:   # up
                matrix = np.roll(matrix, -amt, 0); matrix[-amt:] = 0
            elif d == 2: # down
                matrix = np.roll(matrix, amt, 0); matrix[:amt] = 0
            elif d == 3: # left
                matrix = np.roll(matrix, -amt, 1); matrix[:, -amt:] = 0
            else:        # right
                matrix = np.roll(matrix, amt, 1); matrix[:, :amt] = 0

        elif t == 'rotate':
            angle = random.randint(-5, 5)
            img = Image.fromarray(matrix * 255)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
            matrix = (np.array(img) > 0).astype(np.uint8)

        elif t == 'add_noise':
            for _ in range(random.randint(1, 5)):
                matrix[random.randint(0, 29), random.randint(0, 29)] = 1

        elif t == 'flip':
            idx = np.argwhere(matrix == 1)
            if idx.size:
                r, c = idx[random.randrange(len(idx))]
                matrix[r, c] = 0

    # ---- 生成下一个文件名 ----
    max_num = max((int(f) for f in os.listdir(data) if f.isdigit()), default=-1)
    new_name = str(max_num + 1)
    new_path = os.path.join(data, new_name)

    with open(new_path, 'w') as f:
        f.write(one_hot + '\n')
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')

    return new_name


if __name__ == '__main__':
    # 自动统计样本数量 n
    n = len([f for f in os.listdir(data) if f.isdigit() and f.isascii()])
    if n == 0:
        print("data目录没有样本")
        sys.exit()

    # 执行 10 轮，每轮把 0…n-1 处理一遍
    for round_idx in range(50):
        for i in range(n):
            process_sample(i)
    print("全部处理完成")