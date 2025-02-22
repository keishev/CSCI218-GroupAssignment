import pygame,sys,random, threading, time
from pygame.math import Vector2

class SNAKE:
	def __init__(self):
		self.body = [Vector2(5,10),Vector2(4,10),Vector2(3,10)]
		self.direction = Vector2(0,0)
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

		for index,block in enumerate(self.body):
			x_pos = int(block.x * cell_size)
			y_pos = int(block.y * cell_size)
			block_rect = pygame.Rect(x_pos,y_pos,cell_size,cell_size)

			if index == 0:
				screen.blit(self.head,block_rect)
			elif index == len(self.body) - 1:
				screen.blit(self.tail,block_rect)
			else:
				previous_block = self.body[index + 1] - block
				next_block = self.body[index - 1] - block
				if previous_block.x == next_block.x:
					screen.blit(self.body_vertical,block_rect)
				elif previous_block.y == next_block.y:
					screen.blit(self.body_horizontal,block_rect)
				else:
					if previous_block.x == -1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == -1:
						screen.blit(self.body_tl,block_rect)
					elif previous_block.x == -1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == -1:
						screen.blit(self.body_bl,block_rect)
					elif previous_block.x == 1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == 1:
						screen.blit(self.body_tr,block_rect)
					elif previous_block.x == 1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == 1:
						screen.blit(self.body_br,block_rect)

	def update_head_graphics(self):
		head_relation = self.body[1] - self.body[0]
		if head_relation == Vector2(1,0): self.head = self.head_left
		elif head_relation == Vector2(-1,0): self.head = self.head_right
		elif head_relation == Vector2(0,1): self.head = self.head_up
		elif head_relation == Vector2(0,-1): self.head = self.head_down

	def update_tail_graphics(self):
		tail_relation = self.body[-2] - self.body[-1]
		if tail_relation == Vector2(1,0): self.tail = self.tail_left
		elif tail_relation == Vector2(-1,0): self.tail = self.tail_right
		elif tail_relation == Vector2(0,1): self.tail = self.tail_up
		elif tail_relation == Vector2(0,-1): self.tail = self.tail_down

	def move_snake(self):
		with self.direction_lock:
			direction = self.direction
		if self.new_block == True:
			body_copy = self.body[:]
			body_copy.insert(0,body_copy[0] + self.direction)
			self.body = body_copy[:]
			self.new_block = False
		else:
			body_copy = self.body[:-1]
			body_copy.insert(0,body_copy[0] + self.direction)
			self.body = body_copy[:]

	def add_block(self):
		self.new_block = True

	def play_crunch_sound(self):
		self.crunch_sound.play()

	def reset(self):
		self.body = [Vector2(5,10),Vector2(4,10),Vector2(3,10)]
		self.direction = Vector2(0,0)


class FRUIT:
	def __init__(self):
		self.randomize()

	def draw_fruit(self):
		fruit_rect = pygame.Rect(int(self.pos.x * cell_size),int(self.pos.y * cell_size),cell_size,cell_size)
		screen.blit(apple,fruit_rect)
		#pygame.draw.rect(screen,(126,166,114),fruit_rect)

	def randomize(self):
		self.x = random.randint(0,cell_number - 1)
		self.y = random.randint(0,cell_number - 1)
		self.pos = Vector2(self.x,self.y)

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
		grass_color = (167,209,61)
		for row in range(cell_number):
			if row % 2 == 0:
				for col in range(cell_number):
					if col % 2 == 0:
						grass_rect = pygame.Rect(col * cell_size,row * cell_size,cell_size,cell_size)
						pygame.draw.rect(screen,grass_color,grass_rect)
			else:
				for col in range(cell_number):
					if col % 2 != 0:
						grass_rect = pygame.Rect(col * cell_size,row * cell_size,cell_size,cell_size)
						pygame.draw.rect(screen,grass_color,grass_rect)

	def draw_score(self):
		score_text = str(len(self.snake.body) - 3)
		score_surface = game_font.render(score_text,True,(56,74,12))
		score_x = int(cell_size * cell_number - 60)
		score_y = int(cell_size * cell_number - 40)
		score_rect = score_surface.get_rect(center = (score_x,score_y))
		apple_rect = apple.get_rect(midright = (score_rect.left,score_rect.centery))
		bg_rect = pygame.Rect(apple_rect.left,apple_rect.top,apple_rect.width + score_rect.width + 6,apple_rect.height)

		pygame.draw.rect(screen,(167,209,61),bg_rect)
		screen.blit(score_surface,score_rect)
		screen.blit(apple,apple_rect)
		pygame.draw.rect(screen,(56,74,12),bg_rect,2)

pygame.mixer.pre_init(44100,-16,2,512)
pygame.init()
# Define the screen dimensions
cell_size = 30
cell_number = 20
webcam_width = 640  # Width for the webcam feed
screen_width = (cell_number * cell_size) + webcam_width
screen_height = cell_number * cell_size

screen = pygame.display.set_mode((screen_width, screen_height))

clock = pygame.time.Clock()
apple = pygame.image.load('Graphics/apple.png').convert_alpha()
game_font = pygame.font.Font('Font/PoetsenOne-Regular.ttf', 25)

SCREEN_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SCREEN_UPDATE,250)

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

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    with game_update_lock:
        if game_update_flag:
            main_game.update()
            game_update_flag = False

    screen.fill((175, 215, 70))
    main_game.draw_elements()

    # Draw the webcam feed with YOLO inference
    if shared_frame is not None:
        frame = cv2.resize(shared_frame, (webcam_width, screen_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(frame_surface, (cell_number * cell_size, 0))

    pygame.display.update()
    clock.tick(60)

