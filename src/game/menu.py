import pygame
import sys
from src.config.game_config import *

class Button:
    def __init__(self, x, y, width, height, text, font_size=32):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = pygame.font.Font(None, font_size)
        self.is_hovered = False

    def draw(self, screen):
        color = BUTTON_HOVER_COLOR if self.is_hovered else BUTTON_COLOR
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        
        text_surface = self.font.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                return True
        return False

class MainMenu:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(MENU_WINDOW_SIZE)
        pygame.display.set_caption("游戏主菜单")
        
        # 创建按钮
        button_x = (MENU_WINDOW_SIZE[0] - BUTTON_SIZE[0]) // 2
        self.level1_button = Button(
            button_x,
            80,
            BUTTON_SIZE[0],
            BUTTON_SIZE[1],
            "Level 1 - 经典模式 (7x7)"
        )
        
        self.level2_button = Button(
            button_x,
            150,
            BUTTON_SIZE[0],
            BUTTON_SIZE[1],
            "Level 2 - 怪物模式 (9x9)"
        )
        
        self.level3_button = Button(
            button_x,
            220,
            BUTTON_SIZE[0],
            BUTTON_SIZE[1],
            "Level 3 - 道具模式 (11x11)"
        )
        
        self.quit_button = Button(
            button_x,
            290,
            BUTTON_SIZE[0],
            BUTTON_SIZE[1],
            "退出游戏"
        )
        
        self.title_font = pygame.font.Font(None, 48)
        self.info_font = pygame.font.Font(None, 20)

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if self.level1_button.handle_event(event):
                    return MODE_LEVEL1
                
                if self.level2_button.handle_event(event):
                    return MODE_LEVEL2
                
                if self.level3_button.handle_event(event):
                    return MODE_LEVEL3
                
                if self.quit_button.handle_event(event):
                    pygame.quit()
                    sys.exit()
            
            # 绘制界面
            self.screen.fill(MENU_BG)
            
            # 绘制标题
            title_surface = self.title_font.render("迷宫游戏", True, BLACK)
            title_rect = title_surface.get_rect(
                center=(MENU_WINDOW_SIZE[0]//2, 40)
            )
            self.screen.blit(title_surface, title_rect)
            
            # 绘制按钮
            self.level1_button.draw(self.screen)
            self.level2_button.draw(self.screen)
            self.level3_button.draw(self.screen)
            self.quit_button.draw(self.screen)
            
            # 添加版本信息
            version_info = self.info_font.render("v1.1.0 - 道具和三级难度", True, BLACK)
            self.screen.blit(version_info, (10, MENU_WINDOW_SIZE[1] - 25))
            
            pygame.display.flip() 