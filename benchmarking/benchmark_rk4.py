import pygame
import time, os
from objects import Particle, Spring
from ode_solvers.rk4 import runge_kutta_4th_order

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

FRAMERATE = 600  # framerate * tickrate
TIMESTEP = 1 / FRAMERATE  # 1 / (FRAMERATE / 60)
DIMENSIONS = 2
ZOOM = 1


def main():
    run = True
    clock = pygame.time.Clock()
    tickcounter = 0

    p1 = Particle([0, 100], 10.00, DIMENSIONS, ZOOM, TIMESTEP, 10.00, DARK_GREY)
    p2 = Particle([0, 250], 10.00, DIMENSIONS, ZOOM, TIMESTEP, 10.00, DARK_GREY)

    spring = Spring(p1, p2, length=100, k=5)

    scene = [p1, p2, spring]

    # function that calculates the forces for the
    # current particles in the scene for one timestep
    def add_forces():
        spring.add_forces()

    # control time
    start_time = time.time()

    energy_diff = 0
    last_energy = p1.energy() + p2.energy() + spring.energy()

    while run:
        clock.tick(FRAMERATE)
        WIN.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # update positions
        runge_kutta_4th_order([p1, p2], add_forces)

        # drawing
        for i in scene:
            i.draw(WIN)

        # total energy in scene
        total_energy = p1.energy() + p2.energy() + spring.energy()
        energy_diff += abs(total_energy - last_energy)
        last_energy = total_energy

        if round(tickcounter * TIMESTEP, 2) == 10:
            print(f"Numerical error (accumulated): {energy_diff}")

        # status text
        tickcounter += 1
        status_text_rt = FONT.render(f"Realtime    : {round(time.time() - start_time, 2)}", 1, WHITE)
        status_text_pt = FONT.render(f"Program Time: {round(tickcounter * TIMESTEP, 2)}", 1, WHITE)
        energy_text = FONT.render(f"Total system energy: {total_energy}", 1, WHITE)
        energy_diff_text = FONT.render(f"Numerical error (accumulated): {round(energy_diff , 5)}", 1, WHITE)
        WIN.blit(status_text_rt, (0, 0))
        WIN.blit(status_text_pt, (0, 20))
        WIN.blit(energy_text, (0, 40))
        WIN.blit(energy_diff_text, (0, 60))

        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()
