import threading
import time

# Global variable for direction from YOLO
labelGame = ""

#############################################
# Thread 1: YOLO Classification from Webcam #
#############################################
def yolo_thread():
    import cv2
    import torch
    from ultralytics import YOLO

    global labelGame

    # Determine device and load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = YOLO("game/best.pt")

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = time.time()
    print("YOLO thread running. Press 'q' in the window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Run classification inference on the frame
        results = model.predict(frame, device=device)
        prediction = results[0]
        probs = prediction.probs  # 'Probs' object
        class_names = prediction.names

        # Retrieve top-1 index and confidence
        top1_idx = probs.top1
        top1_conf = probs.top1conf
        top1_label = class_names[top1_idx]
        if top1_conf > 0.8:
            labelGame = top1_label  # Update the global direction label

        # Add the classification info onto the frame
        text = f"{top1_label} ({float(top1_conf):.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # Display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

####################################################
# Thread 2: Pygame Snake Game using labelGame input #
####################################################
def snake_game_thread():
    import pygame
    import sys
    import random

    global labelGame

    pygame.init()
    width, height = 640, 480
    cell_size = 20
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Snake Game Controlled by YOLO")
    clock = pygame.time.Clock()

    # Initial snake settings
    snake_pos = [[width // 2, height // 2]]
    snake_direction = "RIGHT"  # default direction
    # Mapping from YOLO output to game direction (case-insensitive)
    direction_map = {
        "up": "UP",
        "down": "DOWN",
        "stop": "LEFT",
        "stop_inverted": "RIGHT"
    }
    dx, dy = cell_size, 0  # moving right initially

    # Create initial food position
    food_pos = [random.randrange(1, width // cell_size) * cell_size,
                random.randrange(1, height // cell_size) * cell_size]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        # Update snake direction based on the YOLO output if valid.
        # The YOLO thread updates labelGame continuously.
        if labelGame:
            new_dir = direction_map.get(labelGame.lower(), snake_direction)
            # Prevent the snake from reversing directly
            if (snake_direction == "UP" and new_dir != "DOWN") or \
                    (snake_direction == "DOWN" and new_dir != "UP") or \
                    (snake_direction == "LEFT" and new_dir != "RIGHT") or \
                    (snake_direction == "RIGHT" and new_dir != "LEFT"):
                snake_direction = new_dir

        # Update dx, dy based on current snake_direction
        if snake_direction == "UP":
            dx, dy = 0, -cell_size
        elif snake_direction == "DOWN":
            dx, dy = 0, cell_size
        elif snake_direction == "LEFT":
            dx, dy = -cell_size, 0
        elif snake_direction == "RIGHT":
            dx, dy = cell_size, 0

        # Move the snake by adding a new head
        new_head = [snake_pos[0][0] + dx, snake_pos[0][1] + dy]
        snake_pos.insert(0, new_head)

        # Check if food is eaten; if so, spawn new food; otherwise, remove the tail
        if new_head == food_pos:
            food_pos = [random.randrange(1, width // cell_size) * cell_size,
                        random.randrange(1, height // cell_size) * cell_size]
        else:
            snake_pos.pop()

        # Collision detection with boundaries or self
        if (new_head[0] < 0 or new_head[0] >= width or
                new_head[1] < 0 or new_head[1] >= height or
                new_head in snake_pos[1:]):
            print("Game Over!")
            running = False

        # Drawing
        screen.fill((0, 0, 0))
        for pos in snake_pos:
            pygame.draw.rect(screen, (0, 255, 0),
                             pygame.Rect(pos[0], pos[1], cell_size, cell_size))
        pygame.draw.rect(screen, (255, 0, 0),
                         pygame.Rect(food_pos[0], food_pos[1], cell_size, cell_size))
        pygame.display.flip()

        clock.tick(1)  # Control game speed

    pygame.quit()
    sys.exit()

########################
# Main Thread Launcher #
########################
if __name__ == "__main__":
    # Create and start threads for YOLO classification and snake game
    t1 = threading.Thread(target=yolo_thread)
    t2 = threading.Thread(target=snake_game_thread)

    t1.start()
    t2.start()

    t1.join()
    t2.join()
