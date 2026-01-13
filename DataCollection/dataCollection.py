import cv2
import pygame
import numpy as np
import random
import time
import csv
import os
from datetime import datetime
from random import randint

# ======================
# Configuration
# ======================
SESSION_DURATION = 30          # seconds
FPS = 30
POINT_RADIUS = 16
POINT_SPEED = 600              # pixels/sec
CAMERA_INDEX = 0

# ======================
# Output setup
# ======================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("data", exist_ok=True)

video_path = f"data/Domain_B/4/video_{timestamp}.mp4"
csv_path = f"data/Domain_B/4/coords_{timestamp}.csv"

# ======================
# Initialize camera
# ======================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FPS, FPS)

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Camera not accessible")

cam_h, cam_w = frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (cam_w, cam_h))

# ======================
# Initialize pygame
# ======================
pygame.init()
pygame.mouse.set_visible(False)
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_w, screen_h = screen.get_size()
clock = pygame.time.Clock()

# ======================
# Initialize point
# ======================
x = random.uniform(POINT_RADIUS, screen_w - POINT_RADIUS)
y = random.uniform(POINT_RADIUS, screen_h - POINT_RADIUS)

angle = random.uniform(0, 2 * np.pi)
vx = POINT_SPEED * np.cos(angle)
vy = POINT_SPEED * np.sin(angle)
current_color = (0, 255, 100)

# ======================
# CSV logging
# ======================
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame_id", "timestamp", "x", "y"])

# ======================
# Function change color
# ======================
def get_random_color():
    return (randint(50, 255), randint(50, 255), randint(50, 255))

# ======================
# Main loop
# ======================
start_time = time.time()
frame_id = 0
running = True

while running:
    elapsed = time.time() - start_time
    if elapsed >= SESSION_DURATION:
        break

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    # Read camera frame
    ret, cam_frame = cap.read()
    if not ret:
        break

    # Update point position
    dt = 1.0 / FPS
    x += vx * dt
    y += vy * dt

    if x <= POINT_RADIUS or x >= screen_w - POINT_RADIUS:
        vx *= -1
        current_color = get_random_color()
    if y <= POINT_RADIUS or y >= screen_h - POINT_RADIUS:
        vy *= -1
        current_color = get_random_color()

    # Draw
    screen.fill((0, 0, 0))
    pygame.draw.circle(screen, current_color, (int(x), int(y)), POINT_RADIUS)
    pygame.display.flip()

    # Save data
    video_writer.write(cam_frame)
    csv_writer.writerow([frame_id, elapsed, int(x), int(y)])

    frame_id += 1
    clock.tick(FPS)

# ======================
# Cleanup
# ======================
csv_file.close()
video_writer.release()
cap.release()
pygame.quit()

print("Session completed.")
print("Saved:", video_path)
print("Saved:", csv_path)
