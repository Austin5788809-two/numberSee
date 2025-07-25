import pygame
import numpy as np
import os

pygame.init()
cell_size = 16
matrix_size = 50
canvas_size = matrix_size * cell_size

screen = pygame.display.set_mode((canvas_size, canvas_size))
clock = pygame.time.Clock()

def get_matrix(surface):
    # 将pygame表面转换为numpy数组
    arr = pygame.surfarray.array2d(surface)
    mat = np.zeros((matrix_size, matrix_size), dtype=int)

    for i in range(matrix_size):
        for j in range(matrix_size):
            x = j * cell_size + cell_size // 2
            y = i * cell_size + cell_size // 2
            color = arr[x, y] & 0xFF  # 获取像素的亮度值
            mat[i, j] = 1 if color < 128 else 0  # 阈值128区分黑白色

    return mat

def main():
    drawing = False
    screen.fill((255, 255, 255))  # 白色背景

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击
                    drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:  # 按Enter键
                    surface = pygame.display.get_surface().copy()
                    mat = get_matrix(surface)

                    # 保存矩阵到文件
                    with open("input", "w") as f:
                        for row in mat:
                            f.write(" ".join(str(x) for x in row) + "\n")

                    print("矩阵已保存到 input")
                    pygame.quit()
                    return

        # 绘制逻辑
        if drawing:
            mx, my = pygame.mouse.get_pos()
            gx, gy = mx // cell_size, my // cell_size
            if 0 <= gx < matrix_size and 0 <= gy < matrix_size:
                # 绘制黑色方块
                pygame.draw.rect(screen, (0, 0, 0),
                                (gx * cell_size, gy * cell_size, cell_size, cell_size))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()