# import from directory above
import sys
sys.path.append("..")
from objects import Particle, coords_to_pygame
from constraints import ConstraintManager
from ode_solvers.rk4 import runge_kutta_4th_order
import numpy as np
from forces import Gravity, LinearFrictionForce

import time, os, pygame

pygame.init()

WIDTH, HEIGHT = 1000, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(os.path.basename(__file__))

WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (100, 149, 237)
RED = (188, 39, 50)
DARK_GREY = (80, 78, 81)

FONT = pygame.font.SysFont("DejaVu Sans", 20)

FRAMERATE = 160
TIMESTEP = 1 / FRAMERATE
DIMENSIONS = 2
ZOOM = 100


def draw_connection_line(p1: Particle, p2: Particle) -> None:
    pygame.draw.line(WIN, WHITE, coords_to_pygame((ZOOM * p1.position[0], ZOOM * p1.position[1])),
                     coords_to_pygame((ZOOM * p2.position[0], ZOOM * p2.position[1])), 1)


def particles_distance(p1: Particle, p2: Particle) -> float:
    return np.sqrt((p1.position[0] - p2.position[0]) ** 2 + (p1.position[1] - p2.position[1]) ** 2)


def main():
    run = True
    clock = pygame.time.Clock()
    tickcounter = 0

    p1 = Particle([0, 0], 1.00, DIMENSIONS, ZOOM, TIMESTEP, 0.30, YELLOW)
    p2 = Particle([1, 0], 1.00, DIMENSIONS, ZOOM, TIMESTEP, 0.30, DARK_GREY)
    p3 = Particle([1, -1], 1.00, DIMENSIONS, ZOOM, TIMESTEP, 0.30, YELLOW)
    p4 = Particle([1, -2], 1.00, DIMENSIONS, ZOOM, TIMESTEP, 0.30, DARK_GREY)

    scene = [p1, p2, p3, p4]

    # gravity force
    gravity = Gravity([p2, p3, p4], 9.81, 1)  # dimension starting from 0
    linear_friction = LinearFrictionForce([p2, p3, p4], 0.1, DIMENSIONS)

    # import/create constraints
    constraints_scene = [p2, p3, p4]
    constraint_manager = ConstraintManager(constraints_scene, DIMENSIONS)

    def add_forces() -> None:
        gravity.add_forces()
        linear_friction.add_forces()

        constraint_manager.update()
        constraint_manager.circular_wire_constraint(p2)
        constraint_manager.distance_constraint(p2, p3)
        constraint_manager.distance_constraint(p3, p4)
        constraint_manager.add_forces()

    # control time
    start_time = time.time()

    while run:
        clock.tick(FRAMERATE)
        WIN.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # ode solver
        runge_kutta_4th_order([p2, p3, p4],
                              add_forces)

        total_energy = (p1.energy() + p2.energy()
                        + p3.energy() + p4.energy()
                        + gravity.potential_energy())

        #############################
        ###### DRAWING SECTION ######
        #############################

        # drawing the particles themselves
        for i in scene:
            i.draw(WIN)

        # drawing the pendulum connection lines
        draw_connection_line(p1, p2)
        draw_connection_line(p2, p3)
        draw_connection_line(p3, p4)

        # draw each particles mass on the particle itself
        for i in scene:
            mass_text = FONT.render(f"{round(i.mass, 2)}kg", 1, WHITE)
            WIN.blit(mass_text, coords_to_pygame((i.position[0] * ZOOM + 40, i.position[1] * ZOOM + 40)))

        # status texts
        tickcounter += 1
        status_text_rt = FONT.render(f"Realtime    : {round(time.time() - start_time, 2)}", 1, WHITE)
        status_text_pt = FONT.render(f"Program Time: {round(tickcounter * TIMESTEP, 2)}", 1, WHITE)
        energy_text = FONT.render(f"Total system energy: {round(total_energy, 3)} (Numerical Error)", 1, WHITE)
        distance_text = FONT.render(f"Distance between p2 and p3: {round(particles_distance(p2, p3), 3)}", 1, WHITE)
        distance_text2 = FONT.render(f"Distance between p3 and p4: {round(particles_distance(p3, p4), 3)}", 1, WHITE)
        WIN.blit(status_text_rt, (0, 0))
        WIN.blit(status_text_pt, (0, 20))
        WIN.blit(energy_text, (0, 40))
        WIN.blit(distance_text, (0, 60))
        WIN.blit(distance_text2, (0, 80))

        # draw a scale line
        position_legend = [1, 3]
        pygame.draw.line(WIN, WHITE, (1*ZOOM, 3*ZOOM), (2*ZOOM, 3*ZOOM), 1)
        pygame.draw.line(WIN, WHITE, (1*ZOOM, 3.05*ZOOM), (1*ZOOM, 2.95*ZOOM), 1)
        pygame.draw.line(WIN, WHITE, (2*ZOOM, 3.05*ZOOM), (2*ZOOM, 2.95*ZOOM), 1)
        # write the scale value
        scale_text = FONT.render("1m", 1, WHITE)
        WIN.blit(scale_text, (1.35*ZOOM, 3.1*ZOOM))

        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()
