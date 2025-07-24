import pygame
import numpy as np
import os

pygame.init()
cell_size = 16
board_size = 30
canvas_size = board_size * cell_size
matrix_size = 50

screen = pygame.display.set_mode((canvas_size, canvas_size))
clock = pygame.time.Clock()

cnt = 1
os.makedirs("data", exist_ok=True)

def get_matrix(surface):
    arr = pygame.surfarray.array2d(surface)
    mat = np.zeros((matrix_size, matrix_size), dtype=int)
    # 将画板内容映射到中间30*30
    for i in range(board_size):
        for j in range(board_size):
            x = j * cell_size + cell_size // 2
            y = i * cell_size + cell_size // 2
            color = arr[x, y] & 0xFF
            mat[i+2, j+2] = 1 if color < 128 else 0
    return mat

while True:
    file_path = f"data/{cnt}"
    f = open(file_path, "w")
    pygame.display.set_caption(str(cnt % 10))
    screen.fill((255, 255, 255))
    drawing = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    break
        else:
            if drawing:
                mx, my = pygame.mouse.get_pos()
                gx, gy = mx // cell_size, my // cell_size
                if 0 <= gx < board_size and 0 <= gy < board_size:
                    pygame.draw.rect(screen, (0, 0, 0), (gx*cell_size, gy*cell_size, cell_size, cell_size))
            pygame.display.flip()
            clock.tick(60)
            continue
        break

    surface = pygame.display.get_surface().copy()
    mat = get_matrix(surface)
    # 写入文件
    f.write(f"{cnt%10}\n")
    for row in mat:
        f.write(" ".join(str(x) for x in row) + "\n")
    f.close()
    cnt += 1