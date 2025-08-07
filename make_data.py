"""
draw_dataset.py
用鼠标在 20×20 空白画布上画数字 0~9，
按 Enter 保存样本，按 ESC 退出。
保存格式：
第一行：10 维 one-hot 标签（空格分隔）
接下来 30 行：30×30 的 0/1 矩阵（空格分隔）
文件名：data/<filename> （无后缀）
"""

import os
import sys
import numpy as np
import pygame

# -------------------- 基础配置 --------------------
GRID_SIZE   = 20          # 原始画布尺寸
CANVAS_SIZE = 30          # 保存时扩展后的尺寸
CELL_PX     = 20          # 每个格子像素大小
WINDOW_WH   = GRID_SIZE * CELL_PX

filename = 0              # 样本计数
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------- Pygame 初始化 --------------------
pygame.init()
screen = pygame.display.set_mode((WINDOW_WH, WINDOW_WH))
pygame.display.set_caption("Draw Dataset - 0")
clock  = pygame.time.Clock()

# 20×20 的 0/1 矩阵，初始为 0（空白）
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

drawing = False

# -------------------- 工具函数 --------------------
def draw_grid():
    """实时把 grid 画到窗口"""
    screen.fill((255, 255, 255))
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            color = (0, 0, 0) if grid[y, x] else (255, 255, 255)
            rect = pygame.Rect(x*CELL_PX, y*CELL_PX, CELL_PX, CELL_PX)
            screen.fill(color, rect)
    pygame.display.flip()

def expand_and_save(label_int):
    """
    把 20×20 的 grid 上下左右各加 5 格空白，得到 30×30，
    然后按指定格式写到 data/<filename>
    """
    big = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    big[5:5+GRID_SIZE, 5:5+GRID_SIZE] = grid

    one_hot = np.zeros(10, dtype=np.uint8)
    one_hot[label_int] = 1

    path = os.path.join(DATA_DIR, str(filename))
    with open(path, "w") as f:
        # 第一行：one-hot
        f.write(" ".join(map(str, one_hot)) + "\n")
        # 接下来 30 行：0/1 矩阵
        for row in big:
            f.write(" ".join(map(str, row)) + "\n")

# -------------------- 主循环 --------------------
def main():
    global filename, drawing   # 新增 drawing 的 global 声明

    while True:
        for digit in range(10):
            pygame.display.set_caption(f"Draw Dataset - {digit}")
            grid.fill(0)          # 清空网格
            drawing = False       # 每轮开始前重置为 False
            draw_grid()

            # 单幅图采集循环
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit(); sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit(); sys.exit()
                        elif event.key == pygame.K_RETURN:
                            expand_and_save(digit)
                            filename += 1
                            running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        drawing = True
                    elif event.type == pygame.MOUSEBUTTONUP:
                        drawing = False
                    elif event.type == pygame.MOUSEMOTION and drawing:
                        mx, my = event.pos
                        x, y = mx // CELL_PX, my // CELL_PX
                        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                            grid[y, x] = 1
                        draw_grid()
                clock.tick(120)
if __name__ == "__main__":
    main()