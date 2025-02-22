import cv2
import torch
import time
import pygame
import random
from threading import Thread
from pygame.locals import *
from ultralytics import YOLO

# Shared variables between threads
labelGame = "right"  # Initial direction
exit_flag = False

# YOLO Classification Thread
def yolo_classification():
    global labelGame, exit_flag

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("game/best.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit_flag = True
        return

    prev_time = time.time()

    while not exit_flag:
        ret, frame = cap.read()
        if not ret:
            continue

        # Run inference
        results = model.predict(frame, device=device)
        prediction = results[0]

        if prediction.probs:
            probs = prediction.probs
            class_names = prediction.names
            top1_idx = probs.top1
            top1_conf = probs.top1conf
            top1_label = class_names[top1_idx]
            if top1_conf > 0.8:
                labelGame = top1_label  # Update the global direction label

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Display info
        cv2.putText(frame, f"{labelGame} ({probs.top1conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("YOLO Classification", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_flag = True
            break

    cap.release()
    cv2.destroyAllWindows()

# Pygame Snake Game (Main Thread)
class SnakeGame:
    def __init__(self):
        pygame.init()
        self.cell_size = 40
        self.cell_number = 15
        self.screen = pygame.display.set_mode(
            (self.cell_number * self.cell_size,
             self.cell_number * self.cell_size))
        pygame.display.set_caption("YOLO-Controlled Snake")

        self.snake = [(5, 5), (4, 5), (3, 5)]
        self.direction = "right"
        self.food = self.new_food()
        self.score = 0

    def new_food(self):
        while True:
            food = (random.randint(0, self.cell_number-1),
                    random.randint(0, self.cell_number-1))
            if food not in self.snake:
                return food

    def run(self):
        global exit_flag
        clock = pygame.time.Clock()

        while not exit_flag:
            # Handle exit events
            for event in pygame.event.get():
                if event.type == QUIT:
                    exit_flag = True

            # Update direction from YOLO classification
            new_dir = labelGame.lower()
            # map label to new_dir
            direction_map = {
                "up": "up",
                "down": "down",
                "stop": "left",
                "stop_inverted": "right"
            }
            new_dir = direction_map.get(new_dir, new_dir)
            if new_dir in ["up", "down", "left", "right"]:
                if (new_dir == "up" and self.direction != "down" or
                        new_dir == "down" and self.direction != "up" or
                        new_dir == "left" and self.direction != "right" or
                        new_dir == "right" and self.direction != "left"):
                    self.direction = new_dir

            # Move snake
            head_x, head_y = self.snake[0]
            new_head = {
                "up": (head_x, head_y - 1),
                "down": (head_x, head_y + 1),
                "left": (head_x - 1, head_y),
                "right": (head_x + 1, head_y)
            }[self.direction]

            # Check collisions
            if (new_head[0] < 0 or new_head[0] >= self.cell_number or
                    new_head[1] < 0 or new_head[1] >= self.cell_number or
                    new_head in self.snake):
                exit_flag = True
                break

            self.snake.insert(0, new_head)

            # Food consumption
            if new_head == self.food:
                self.score += 1
                self.food = self.new_food()
            else:
                self.snake.pop()

            # Render
            self.screen.fill((0, 0, 0))

            # Draw snake
            for segment in self.snake:
                pygame.draw.rect(self.screen, (0, 255, 0),
                                 (segment[0]*self.cell_size, segment[1]*self.cell_size,
                                  self.cell_size-2, self.cell_size-2))

            # Draw food
            pygame.draw.rect(self.screen, (255, 0, 0),
                             (self.food[0]*self.cell_size, self.food[1]*self.cell_size,
                              self.cell_size-2, self.cell_size-2))

            pygame.display.update()
            clock.tick(1)  # Game speed

        pygame.quit()

if __name__ == "__main__":
    # Start YOLO thread
    yolo_thread = Thread(target=yolo_classification)
    yolo_thread.start()

    # Start game in main thread
    game = SnakeGame()
    game.run()

    # Wait for YOLO thread to finish
    yolo_thread.join()
    print("Game exited cleanly")