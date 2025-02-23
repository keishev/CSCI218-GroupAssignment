import os

import pygame, sys, random, threading, time
from pygame.math import Vector2

START, PLAYING, LOADING, TUTORIAL = "START", "PLAYING", "LOADING", "TUTORIAL"
game_state = LOADING
loading_duration = 5
loading_start_time = pygame.time.get_ticks() / 1000
pygame.init()

title_font = pygame.font.SysFont("Comic Sans MS", 50, bold=True)
instr_font = pygame.font.SysFont("Arial", 30)


class SNAKE:
    def __init__(self):
        self.body = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10)]
        self.direction = Vector2(0, 0)
        self.new_block = False
        self.direction_lock = threading.Lock()

        self.head_up = pygame.image.load('Graphics/head_up.png').convert_alpha()
        self.head_down = pygame.image.load('Graphics/head_down.png').convert_alpha()
        self.head_right = pygame.image.load('Graphics/head_right.png').convert_alpha()
        self.head_left = pygame.image.load('Graphics/head_left.png').convert_alpha()

        self.tail_up = pygame.image.load('Graphics/tail_up.png').convert_alpha()
        self.tail_down = pygame.image.load('Graphics/tail_down.png').convert_alpha()
        self.tail_right = pygame.image.load('Graphics/tail_right.png').convert_alpha()
        self.tail_left = pygame.image.load('Graphics/tail_left.png').convert_alpha()

        self.body_vertical = pygame.image.load('Graphics/body_vertical.png').convert_alpha()
        self.body_horizontal = pygame.image.load('Graphics/body_horizontal.png').convert_alpha()

        self.body_tr = pygame.image.load('Graphics/body_tr.png').convert_alpha()
        self.body_tl = pygame.image.load('Graphics/body_tl.png').convert_alpha()
        self.body_br = pygame.image.load('Graphics/body_br.png').convert_alpha()
        self.body_bl = pygame.image.load('Graphics/body_bl.png').convert_alpha()
        self.crunch_sound = pygame.mixer.Sound('Sound/crunch.wav')

    def draw_snake(self):
        self.update_head_graphics()
        self.update_tail_graphics()

        for index, block in enumerate(self.body):
            x_pos = int(block.x * cell_size)
            y_pos = int(block.y * cell_size)
            block_rect = pygame.Rect(x_pos, y_pos, cell_size, cell_size)

            if index == 0:
                screen.blit(self.head, block_rect)
            elif index == len(self.body) - 1:
                screen.blit(self.tail, block_rect)
            else:
                previous_block = self.body[index + 1] - block
                next_block = self.body[index - 1] - block
                if previous_block.x == next_block.x:
                    screen.blit(self.body_vertical, block_rect)
                elif previous_block.y == next_block.y:
                    screen.blit(self.body_horizontal, block_rect)
                else:
                    if previous_block.x == -1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == -1:
                        screen.blit(self.body_tl, block_rect)
                    elif previous_block.x == -1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == -1:
                        screen.blit(self.body_bl, block_rect)
                    elif previous_block.x == 1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == 1:
                        screen.blit(self.body_tr, block_rect)
                    elif previous_block.x == 1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == 1:
                        screen.blit(self.body_br, block_rect)

    def update_head_graphics(self):
        head_relation = self.body[1] - self.body[0]
        if head_relation == Vector2(1, 0):
            self.head = self.head_left
        elif head_relation == Vector2(-1, 0):
            self.head = self.head_right
        elif head_relation == Vector2(0, 1):
            self.head = self.head_up
        elif head_relation == Vector2(0, -1):
            self.head = self.head_down

    def update_tail_graphics(self):
        tail_relation = self.body[-2] - self.body[-1]
        if tail_relation == Vector2(1, 0):
            self.tail = self.tail_left
        elif tail_relation == Vector2(-1, 0):
            self.tail = self.tail_right
        elif tail_relation == Vector2(0, 1):
            self.tail = self.tail_up
        elif tail_relation == Vector2(0, -1):
            self.tail = self.tail_down

    def move_snake(self):
        with self.direction_lock:
            direction = self.direction
        if self.new_block == True:
            body_copy = self.body[:]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy[:]
            self.new_block = False
        else:
            body_copy = self.body[:-1]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy[:]

    def add_block(self):
        self.new_block = True

    def play_crunch_sound(self):
        self.crunch_sound.play()

    def reset(self):
        self.body = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10)]
        self.direction = Vector2(0, 0)


class FRUIT:
    def __init__(self):
        self.randomize()

    def draw_fruit(self):
        fruit_rect = pygame.Rect(int(self.pos.x * cell_size), int(self.pos.y * cell_size), cell_size, cell_size)
        screen.blit(apple, fruit_rect)
        # pygame.draw.rect(screen,(126,166,114),fruit_rect)

    def randomize(self):
        self.x = random.randint(0, cell_number - 1)
        self.y = random.randint(0, cell_number - 1)
        self.pos = Vector2(self.x, self.y)


class Screen:
    def __init__(self, screen, game_font):
        self.screen = screen
        self.game_font = game_font

    def draw_start_screen(self):
        self.screen.fill((50, 50, 50))  # Dark background
        title_text = self.game_font.render("Snake Game with Hand Gesture Control", True, (255, 255, 255))
        instruction_text = self.game_font.render("Press SPACE to Start", True, (255, 255, 255))

        title_rect = title_text.get_rect(center=(screen_width // 2, screen_height // 3))
        instruction_rect = instruction_text.get_rect(center=(screen_width // 2, screen_height // 2))

        self.screen.blit(title_text, title_rect)
        self.screen.blit(instruction_text, instruction_rect)

    def draw_loading_screen(self):
        self.screen.fill((50, 50, 50))
        loading_text = self.game_font.render("Loading Camera...", True, (255, 255, 255))
        loading_rect = loading_text.get_rect(center=(screen_width // 2, screen_height // 2))
        self.screen.blit(loading_text, loading_rect)

    def draw_tutorial_screen(self):
        start_y = 90
        instruction_spacing = 120

        self.screen.fill((60, 70, 90))

        title_text = game_font.render("Tutorial", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(screen_width // 2, screen_height // 10))
        screen.blit(title_text, title_rect)

        image_paths = {
            "UP": os.path.join("Graphics", "thumb-up.png"),
            "DOWN": os.path.join("Graphics", "thumb-down.png"),
            "LEFT": os.path.join("Graphics", "stop-symbol.png"),
            "RIGHT": os.path.join("Graphics", "palm-of-hand.png")
        }

        gesture_images = {direction: pygame.transform.scale(pygame.image.load(path), (80, 80))
                          for direction, path in image_paths.items()}

        instructions = [
            ("UP", "Thumbs-up"),
            ("DOWN", "Thumbs-down"),
            ("LEFT", "STOP"),
            ("RIGHT", "Inverted-Stop")
        ]

        font = pygame.font.SysFont("Arial", 28)

        for i, (direction, description) in enumerate(instructions):

            text_surface = font.render(f"{direction}: {description}", True, (230, 230, 230))
            text_rect = text_surface.get_rect(topleft=(200, start_y + i * instruction_spacing))
            screen.blit(text_surface, text_rect)

            if direction in gesture_images:
                image = gesture_images[direction]
                image_rect = image.get_rect(topleft=(100, start_y + i * instruction_spacing - 20))
                screen.blit(image, image_rect)

        prompt_text = font.render("Press SPACE to continue", True, (255, 255, 255))
        prompt_text_rect = prompt_text.get_rect(center=(screen_width // 2, screen_height - 80))
        screen.blit(prompt_text, prompt_text_rect)

        pygame.display.flip()


class MAIN:
    def __init__(self):
        self.snake = SNAKE()
        self.fruit = FRUIT()

    def update(self):
        self.snake.move_snake()
        self.check_collision()
        self.check_fail()

    def draw_elements(self):
        self.draw_grass()
        self.fruit.draw_fruit()
        self.snake.draw_snake()
        self.draw_score()

    def check_collision(self):
        if self.fruit.pos == self.snake.body[0]:
            self.fruit.randomize()
            self.snake.add_block()
            self.snake.play_crunch_sound()

        for block in self.snake.body[1:]:
            if block == self.fruit.pos:
                self.fruit.randomize()

    def check_fail(self):
        if not 0 <= self.snake.body[0].x < cell_number or not 0 <= self.snake.body[0].y < cell_number:
            self.game_over()

        for block in self.snake.body[1:]:
            if block == self.snake.body[0]:
                self.game_over()

    def game_over(self):
        self.snake.reset()

    def draw_grass(self):
        grass_color = (167, 209, 61)
        for row in range(cell_number):
            if row % 2 == 0:
                for col in range(cell_number):
                    if col % 2 == 0:
                        grass_rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                        pygame.draw.rect(screen, grass_color, grass_rect)
            else:
                for col in range(cell_number):
                    if col % 2 != 0:
                        grass_rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                        pygame.draw.rect(screen, grass_color, grass_rect)

    def draw_score(self):
        score_text = str(len(self.snake.body) - 3)
        score_surface = game_font.render(score_text, True, (56, 74, 12))
        score_x = int(cell_size * cell_number - 60)
        score_y = int(cell_size * cell_number - 40)
        score_rect = score_surface.get_rect(center=(score_x, score_y))
        apple_rect = apple.get_rect(midright=(score_rect.left, score_rect.centery))
        bg_rect = pygame.Rect(apple_rect.left, apple_rect.top, apple_rect.width + score_rect.width + 6,
                              apple_rect.height)

        pygame.draw.rect(screen, (167, 209, 61), bg_rect)
        screen.blit(score_surface, score_rect)
        screen.blit(apple, apple_rect)
        pygame.draw.rect(screen, (56, 74, 12), bg_rect, 2)


pygame.mixer.pre_init(44100, -16, 2, 512)

cell_size = 30
cell_number = 20
webcam_width = 640
webcam_height = 480
screen_width = (cell_number * cell_size) + webcam_width
screen_height = cell_number * cell_size

screen = pygame.display.set_mode((screen_width, screen_height))

clock = pygame.time.Clock()
apple = pygame.image.load('Graphics/apple.png').convert_alpha()
game_font = pygame.font.Font('Font/PoetsenOne-Regular.ttf', 25)

SCREEN_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SCREEN_UPDATE, 250)

main_game = MAIN()

# Game speed control
game_update_interval = 0.3  # 300ms (slower than original)
game_update_flag = False
game_update_lock = threading.Lock()


def game_speed_controller():
    global game_update_flag
    while True:
        time.sleep(game_update_interval)
        with game_update_lock:
            game_update_flag = True


game_speed_thread = threading.Thread(target=game_speed_controller, daemon=True)
game_speed_thread.start()

# Input handling using keyboard module
import cv2

cap = cv2.VideoCapture(0)


def input_handler():
    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = YOLO("best.pt")

    global shared_frame
    prev_time = time.time()
    print("YOLO thread running.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        results = model.predict(frame, device=device)
        prediction = results[0]
        probs = prediction.probs
        class_names = prediction.names

        top1_idx = probs.top1
        top1_conf = probs.top1conf
        top1_label = class_names[top1_idx]

        if top1_conf > 0.8:
            with main_game.snake.direction_lock:
                if top1_label == "like" and main_game.snake.direction.y != 1:
                    main_game.snake.direction = Vector2(0, -1)
                elif top1_label == "dislike" and main_game.snake.direction.y != -1:
                    main_game.snake.direction = Vector2(0, 1)
                elif top1_label == "stop" and main_game.snake.direction.x != 1:
                    main_game.snake.direction = Vector2(-1, 0)
                elif top1_label == "stop_inverted" and main_game.snake.direction.x != -1:
                    main_game.snake.direction = Vector2(1, 0)

        text = f"{top1_label} ({float(top1_conf):.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        with frame_lock:
            shared_frame = frame.copy()


input_thread = threading.Thread(target=input_handler, daemon=True)
input_thread.start()

shared_frame = None
frame_lock = threading.Lock()
screen_manager = Screen(screen, game_font)

while True:
    ok_button_rect = pygame.Rect(350, 500, 100, 50)

    # for event in pygame.event.get():
    #     if event.type == pygame.QUIT:
    #         pygame.quit()
    #         sys.exit()
    #     elif event.type == pygame.KEYDOWN:
    #         if game_state == START and event.key == pygame.K_SPACE:
    #             game_state = PLAYING
    #             print("Game state changed to PLAYING")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if game_state == TUTORIAL:
            # For example, if user presses Enter or clicks an OK button, start the game
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                game_state = PLAYING
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Assume you have an ok_button_rect defined in your tutorial screen
                mouse_pos = pygame.mouse.get_pos()
                if ok_button_rect.collidepoint(mouse_pos):
                    game_state = PLAYING

    if game_state == LOADING:
        screen_manager.draw_loading_screen()
        current_time = pygame.time.get_ticks() / 1000
        if current_time - loading_start_time >= loading_duration:
            # game_state = START
            game_state = TUTORIAL

    elif game_state == TUTORIAL:
        screen.fill((50, 50, 50))
        screen_manager.draw_tutorial_screen()

    elif game_state == START:
        screen.fill((50, 50, 50))
        screen_manager.draw_start_screen()

    elif game_state == PLAYING:
        screen.fill((175, 215, 70))
        with game_update_lock:
            if game_update_flag:
                main_game.update()
                game_update_flag = False

        main_game.draw_elements()

        with frame_lock:
            if shared_frame is not None:
                try:
                    frame = cv2.resize(shared_frame, (webcam_width, webcam_height))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    screen.blit(frame_surface, (cell_number * cell_size, 0))
                except Exception as e:
                    print(f"Error displaying webcam feed: {e}")

        if not 0 <= main_game.snake.body[0].x < cell_number or not 0 <= main_game.snake.body[0].y < cell_number:
            print("Collision with wall. Resetting game...")
            main_game.snake.reset()
            game_state = START

        for block in main_game.snake.body[1:]:
            if block == main_game.snake.body[0]:
                print("Collision with self. Resetting game...")
                main_game.snake.reset()
                game_state = START

    pygame.display.update()
    clock.tick(60)

