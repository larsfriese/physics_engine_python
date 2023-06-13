import time, os, math, pygame

from objects import Object, Gravity, coords_to_pygame
from constraints import ConstraintManager
from ode_solvers import rk4

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

FRAMERATE = 400
TIMESTEP = 1 / FRAMERATE


def main():
    run = True
    clock = pygame.time.Clock()
    tickcounter = 0

    p1 = Object(0, 0, 10, YELLOW, 1.00, TIMESTEP)
    p1.x_vel = 0
    p1.y_vel = 0

    p2 = Object(100, 0, 10, DARK_GREY, 2.00, TIMESTEP)
    p2.y_vel = 0
    p2.y_vel = 0

    p3 = Object(100, -100, 10, YELLOW, 2.00, TIMESTEP)
    p3.y_vel = 0
    p3.y_vel = 0

    p4 = Object(100, -200, 10, DARK_GREY, 2.00, TIMESTEP)
    p4.y_vel = 0
    p4.y_vel = 0

    scene = [p1, p2, p3, p4]

    # particles/objects used for the constraints
    constraints_scene = [p2, p3, p4]

    # gravity force
    gravity = Gravity([p2, p3, p4], 981)

    # import/create constraints
    constraint_manager = ConstraintManager(constraints_scene)

    def add_forces():
        # add gravity
        gravity.add_forces()

        # update the state variables
        constraint_manager.update(constraints_scene)

        # calculate/add constraint forces
        constraint_manager.circular_wire_constraint(p2)
        # could also be distance constraint between p1 and p2,
        # but this circular wire constraint is less computationally expensive
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
        rk4([p2, p3, p4], add_forces)

        # drawing
        for i in scene:
            i.draw(WIN)

        # pendulum connection line
        pygame.draw.line(WIN, WHITE, coords_to_pygame((p1.x, p1.y)), coords_to_pygame((p2.x, p2.y)), 1)
        pygame.draw.line(WIN, WHITE, coords_to_pygame((p2.x, p2.y)), coords_to_pygame((p3.x, p3.y)), 1)
        pygame.draw.line(WIN, WHITE, coords_to_pygame((p3.x, p3.y)), coords_to_pygame((p4.x, p4.y)), 1)

        # total energy in scene
        total_energy = p1.energy() + p2.energy() + p3.energy() + p4.energy() + gravity.potential_energy()

        # status text
        tickcounter += 1
        status_text_rt = FONT.render(f"Realtime    : {round(time.time() - start_time, 2)}", 1, WHITE)
        status_text_pt = FONT.render(f"Program Time: {round(tickcounter * TIMESTEP, 2)}", 1, WHITE)
        energy_text = FONT.render(f"Total system energy: {round(total_energy, 2)} (Numerical Error)", 1, WHITE)
        distance_text = FONT.render(f"Distance between p2 and p3: {round(math.sqrt((p2.x - p3.x) ** 2 + (p2.y - p3.y) ** 2), 2)}", 1, WHITE)
        distance_text2 = FONT.render(f"Distance between p3 and p4: {round(math.sqrt((p3.x - p4.x) ** 2 + (p3.y - p4.y) ** 2), 2)}", 1, WHITE)
        WIN.blit(status_text_rt, (0, 0))
        WIN.blit(status_text_pt, (0, 20))
        WIN.blit(energy_text, (0, 40))
        WIN.blit(distance_text, (0, 60))
        WIN.blit(distance_text2, (0, 80))

        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()
