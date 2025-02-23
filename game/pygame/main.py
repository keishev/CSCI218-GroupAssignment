import pygame
import sys
import random
import asyncio
import websockets
import json
import cv2
import threading

# Initialize Pygame
pygame.init()

# Screen settings
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
GRID_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Directions
UP = 'UP'
DOWN = 'DOWN'
LEFT = 'LEFT'
RIGHT = 'RIGHT'

# Gesture to direction mapping
gesture_to_direction = {
    "like": UP,
    "dislike": DOWN,
    "fist": LEFT,
    "ok": RIGHT
}

class SnakeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Hand Gesture Controlled Snake Game')
        self.clock = pygame.time.Clock()
        self.reset()
        self.webcam_frame = None
        self.webcam_thread = threading.Thread(target=self.start_webcam_feed)
        self.webcam_thread.daemon = True
        self.webcam_thread.start()

    def reset(self):
        self.snake = [(5, 5)]
        self.food = self.get_random_position()
        self.direction = RIGHT
        self.score = 0
        self.is_game_active = False
        self.game_over = False

    def get_random_position(self):
        return (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))

    def draw_grid(self):
        for x in range(0, SCREEN_WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, WHITE, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, WHITE, (0, y), (SCREEN_WIDTH, y))

    def draw_snake(self):
        for segment in self.snake:
            rect = pygame.Rect(segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(self.screen, GREEN, rect)

    def draw_food(self):
        rect = pygame.Rect(self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(self.screen, RED, rect)

    def move_snake(self):
        head = self.snake[0]
        if self.direction == UP:
            new_head = (head[0], head[1] - 1)
        elif self.direction == DOWN:
            new_head = (head[0], head[1] + 1)
        elif self.direction == LEFT:
            new_head = (head[0] - 1, head[1])
        elif self.direction == RIGHT:
            new_head = (head[0] + 1, head[1])

        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT or
            new_head in self.snake):
            self.game_over = True
            return

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.food = self.get_random_position()
            self.score += 10
        else:
            self.snake.pop()

    def display_score(self):
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (10, 10))

    async def handle_gesture_input(self):
        uri = 'ws://localhost:5003/start-camera'
        async with websockets.connect(uri) as websocket:
            while not self.game_over:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    gesture = data.get('gesture')
                    if gesture in gesture_to_direction:
                        self.direction = gesture_to_direction[gesture]
                except Exception as e:
                    print('WebSocket error:', e)

    def start_webcam_feed(self):
        cap = cv2.VideoCapture('http://localhost:5002/video_feed')
        while not self.game_over:
            ret, frame = cap.read()
            if ret:
                self.webcam_frame = cv2.resize(frame, (200, 150))
                cv2.waitKey(1)
        cap.release()

    def draw_webcam_feed(self):
        if self.webcam_frame is not None:
            frame = cv2.cvtColor(self.webcam_frame, cv2.COLOR_BGR2RGB)
            frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            self.screen.blit(frame, (SCREEN_WIDTH - 210, 10))

    async def game_loop(self):
        self.is_game_active = True
        gesture_task = asyncio.create_task(self.handle_gesture_input())
        while not self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.screen.fill(BLACK)
            self.draw_grid()
            self.draw_snake()
            self.draw_food()
            self.display_score()
            self.draw_webcam_feed()
            self.move_snake()
            pygame.display.flip()
            self.clock.tick(10)
        await gesture_task

if __name__ == '__main__':
    game = SnakeGame()
    asyncio.run(game.game_loop())
