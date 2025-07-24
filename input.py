import os
import pygame
import sys

pygame.init()

# ----------------- 参数 -----------------
GRID_SIZE   = 10
CELL_SIZE   = 40
BTN_W, BTN_H = 120, 50
MARGIN      = 20
WIN_W  = GRID_SIZE * CELL_SIZE + 2 * MARGIN
WIN_H  = GRID_SIZE * CELL_SIZE + 2 * MARGIN + BTN_H + 10

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY  = (200, 200, 200)

FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
font      = pygame.font.Font(FONT_PATH, 24)
smallfont = pygame.font.Font(FONT_PATH, 20)

screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("绘图网格")

grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
drawing = False

confirm_btn = pygame.Rect(MARGIN, MARGIN + GRID_SIZE * CELL_SIZE + 10, BTN_W, BTN_H)
quit_btn    = pygame.Rect(WIN_W - MARGIN - BTN_W, MARGIN + GRID_SIZE * CELL_SIZE + 10, BTN_W, BTN_H)

# =====================================================================
# 1. 程序启动时先让用户输入“起始文件编号 n”
# =====================================================================
def ask_start_number():
    """返回用户输入的起始编号（int），允许 0 及以上"""
    input_rect = pygame.Rect(WIN_W//2 - 70, WIN_H//2 - 20, 140, 40)
    color = pygame.Color('dodgerblue2')
    text  = ""
    prompt = smallfont.render("请输入起始文件编号（≥0）：", True, BLACK)
    prompt_rect = prompt.get_rect(center=(WIN_W//2, WIN_H//2 - 60))
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and text.isdigit():
                    return int(text)
                elif event.key == pygame.K_BACKSPACE:
                    text = text[:-1]
                elif event.unicode.isdigit():
                    text += event.unicode

        screen.fill(WHITE)
        screen.blit(prompt, prompt_rect)
        txt_surf = smallfont.render(text, True, BLACK)
        screen.blit(txt_surf, (input_rect.x + 5, input_rect.y + 5))
        pygame.draw.rect(screen, color, input_rect, 2)
        pygame.display.flip()
        clock.tick(30)

# 让用户先输入起始编号
start_n = ask_start_number()

# =====================================================================
# 2. 根据起始编号生成下一个文件名
#    例如 start_n = 5，则第一次保存为 data/5，第二次 data/6 …
# =====================================================================
def next_filename():
    """按 start_n 开始返回递增的文件名"""
    global start_n
    os.makedirs("data", exist_ok=True)
    filename = os.path.join("data", str(start_n))
    start_n += 1                 # 每调用一次自增 1
    return filename

# ----------------- 其余函数与主循环 -----------------
def draw_grid():
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            color = BLACK if grid[y][x] else WHITE
            rect = pygame.Rect(MARGIN + x * CELL_SIZE,
                               MARGIN + y * CELL_SIZE,
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRAY, rect, 1)

def draw_buttons():
    pygame.draw.rect(screen, GRAY, confirm_btn)
    txt = font.render("确认", True, BLACK)
    screen.blit(txt, txt.get_rect(center=confirm_btn.center))

    pygame.draw.rect(screen, GRAY, quit_btn)
    txt = font.render("结束", True, BLACK)
    screen.blit(txt, txt.get_rect(center=quit_btn.center))

def save_data(n):
    fname = next_filename()
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        for row in grid:
            f.write("".join(map(str, row)) + "\n")
    print(f"已保存 {fname}")

def input_number_loop():
    input_rect = pygame.Rect(WIN_W//2 - 70, WIN_H//2 - 20, 140, 40)
    color = pygame.Color('dodgerblue2')
    text  = ""
    prompt = smallfont.render("请输入 0~9 的数字：", True, BLACK)
    prompt_rect = prompt.get_rect(center=(WIN_W//2, WIN_H//2 - 60))
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and text.isdigit():
                    n = int(text)
                    if 0 <= n <= 9:
                        save_data(n)
                        return True
                elif event.key == pygame.K_BACKSPACE:
                    text = text[:-1]
                elif event.unicode.isdigit() and len(text) < 1:
                    text += event.unicode
        screen.fill(WHITE)
        screen.blit(prompt, prompt_rect)
        txt_surf = smallfont.render(text, True, BLACK)
        screen.blit(txt_surf, (input_rect.x + 5, input_rect.y + 5))
        pygame.draw.rect(screen, color, input_rect, 2)
        pygame.display.flip()
        clock.tick(30)

# ----------------- 主循环 -----------------
clock = pygame.time.Clock()
while True:
    screen.fill(WHITE)
    draw_grid()
    draw_buttons()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if confirm_btn.collidepoint(mx, my):
                if input_number_loop():
                    grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
            elif quit_btn.collidepoint(mx, my):
                pygame.quit(); sys.exit()
            else:
                gx = (mx - MARGIN) // CELL_SIZE
                gy = (my - MARGIN) // CELL_SIZE
                if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                    grid[gy][gx] ^= 1
                    drawing = True
        elif event.type == pygame.MOUSEMOTION and drawing:
            mx, my = event.pos
            gx = (mx - MARGIN) // CELL_SIZE
            gy = (my - MARGIN) // CELL_SIZE
            if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                grid[gy][gx] = 1
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
    pygame.display.flip()
    clock.tick(60)