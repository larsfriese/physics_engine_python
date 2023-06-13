import pygame
import time, os
from objects import Object, Spring, Gravity

pygame.init()

WIDTH, HEIGHT = 1600, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(os.path.basename(__file__))

WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (100, 149, 237)
RED = (188, 39, 50)
DARK_GREY = (80, 78, 81)

FONT = pygame.font.SysFont("DejaVu Sans", 20)

FRAMERATE = 60  # framerate * tickrate
TIMESTEP = 1 / FRAMERATE


def main():
    run = True
    clock = pygame.time.Clock()
    tickcounter = 0

    p1 = Object(300, 300, 10, BLUE, 10, TIMESTEP)
    p1.x_vel = 0
    p1.y_vel = 0

    p2 = Object(200, 300, 10, RED, 10, TIMESTEP)
    p2.y_vel = 0
    p2.y_vel = 0

    p3 = Object(100, 300, 10, RED, 10, TIMESTEP)
    p3.y_vel = 0
    p3.y_vel = 0

    spring = Spring(p1, p2, length=100, k=1000)
    spring2 = Spring(p2, p3, length=100, k=1000)

    scene = [p1, p2, p3, spring, spring2]

    # gravity force
    gravity = Gravity([p2, p3], 9.81)

    # control time
    start_time = time.time()

    while run:
        clock.tick(FRAMERATE)
        WIN.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # preparation for verlet
        p1_verlet = p2.velocity_verlet_1()
        p2_verlet = p3.velocity_verlet_1()

        # actual simulation
        spring.add_forces()
        spring2.add_forces()
        gravity.add_forces()

        # ode
        p2.velocity_verlet_2(p1_verlet)
        p3.velocity_verlet_2(p2_verlet)

        # drawing
        for i in scene:
            i.draw(WIN)

        # total energy in scene
        total_energy = p1.energy() + p2.energy() + spring.energy() + spring2.energy() + gravity.potential_energy()

        # status text
        tickcounter += 1
        status_text_rt = FONT.render(f"Realtime    : {round(time.time() - start_time, 2)}", 1, WHITE)
        status_text_pt = FONT.render(f"Program Time: {round(tickcounter * TIMESTEP, 2)}", 1, WHITE)
        energy_text = FONT.render(f"Total system energy: {round(total_energy, 2)} (Numerical Error)", 1, WHITE)
        WIN.blit(status_text_rt, (0, 0))
        WIN.blit(status_text_pt, (0, 20))
        WIN.blit(energy_text, (0, 40))

        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()
