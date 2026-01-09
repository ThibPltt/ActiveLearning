import pygame
from random import randint

# Initialisation de Pygame
pygame.init()

# Paramètres
SIZE = width, height = 800, 600
BG_COLOR = (0, 0, 0)
RADIUS = 20
FPS = 25

screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption('Cercle Rebondissant avec Log')
clock = pygame.time.Clock()

x = randint(RADIUS, width - RADIUS)
y = randint(RADIUS, height - RADIUS)
x_speed, y_speed = 4, 4
current_color = (0, 255, 100)

def get_random_color():
    return (randint(50, 255), randint(50, 255), randint(50, 255))

exit = False

while not exit:
    screen.fill(BG_COLOR)

    # Rebond et changement de couleur
    if (x + RADIUS >= width) or (x - RADIUS <= 0):
        x_speed = -x_speed
        current_color = get_random_color()
    
    if (y + RADIUS >= height) or (y - RADIUS <= 0):
        y_speed = -y_speed
        current_color = get_random_color()

    x += x_speed
    y += y_speed

    # --- AFFICHAGE DANS LA CONSOLE ---
    # On utilise f-string pour afficher x et y proprement
    print(f"Position du cercle -> x: {int(x)}, y: {int(y)}")

    # Dessin
    pygame.draw.circle(screen, current_color, (int(x), int(y)), RADIUS)

    pygame.display.update()
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit = True

pygame.quit()