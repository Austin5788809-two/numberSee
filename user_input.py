# user_input.py
import numpy as np
import pygame
import sys

GRID = 30          # 画布格子数
CELL = 20          # 每格像素大小
WIN  = GRID * CELL

pygame.init()
screen = pygame.display.set_mode((WIN, WIN))
pygame.display.set_caption("press Enter to finish")
clock = pygame.time.Clock()

grid = np.zeros((GRID, GRID), dtype=np.uint8)   # 0/1 矩阵
drawing = False

def draw_canvas():
    screen.fill((255, 255, 255))
    for y in range(GRID):
        for x in range(GRID):
            if grid[y, x]:
                rect = pygame.Rect(x * CELL, y * CELL, CELL, CELL)
                pygame.draw.rect(screen, (0, 0, 0), rect)
    pygame.display.flip()

# 主循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
            # 保存
            with open("user_input", "w") as f:
                for row in grid:
                    f.write(" ".join(map(str, row)) + "\n")
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION and drawing:
            mx, my = event.pos
            x, y = mx // CELL, my // CELL
            if 0 <= x < GRID and 0 <= y < GRID:
                grid[y, x] = 1
    draw_canvas()
    clock.tick(120)

pygame.quit()
sys.exit()