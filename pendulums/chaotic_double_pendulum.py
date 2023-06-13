import time, os, pygame

from objects import Object, Gravity, coords_to_pygame
from constraints import ConstraintManager
from ode_solvers import rk4

pygame.init()

WIDTH, HEIGHT = 1920, 1080
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(os.path.basename(__file__))

WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (100, 149, 237)
RED = (188, 39, 50)
DARK_GREY = (80, 78, 81)

FONT = pygame.font.SysFont("DejaVu Sans", 20)

FRAMERATE = 60
TIMESTEP = 1 / FRAMERATE


def main():
    run = True
    clock = pygame.time.Clock()
    tickcounter = 0

    objects_p2 = []
    objects_p3 = []
    colors = []

    for i in range(20):
        COLOR = (255, i*10, i*10)
        colors.append(COLOR)

        p2 = Object(100, 0, 10, COLOR, 200, TIMESTEP)
        p2.x_vel = 0
        p2.y_vel = 0

        p3 = Object(100, -100, 10, COLOR, 200, TIMESTEP)
        p3.x_vel = 5*i
        p3.y_vel = 0

        objects_p2.append(p2)
        objects_p3.append(p3)

    p1 = Object(0, 0, 10, YELLOW, 100, TIMESTEP)
    p1.x_vel = 0
    p1.y_vel = 0

    scene = [p1] + objects_p2 + objects_p3

    # particles/objects used for the constraints
    constraints_scene = objects_p2 + objects_p3

    # gravity force
    gravity = Gravity(constraints_scene, 981)

    # import/create constraints
    constraint_manager = ConstraintManager(constraints_scene)

    def add_forces():
        # add gravity
        gravity.add_forces()

        # update the state variables
        constraint_manager.update(constraints_scene)

        # calculate/add constraint forces
        for i in range(len(objects_p2)):
            constraint_manager.circular_wire_constraint(objects_p2[i])
            constraint_manager.distance_constraint(objects_p2[i], objects_p3[i])

        constraint_manager.add_forces()

    # control time
    start_time = time.time()

    while run:
        clock.tick(FRAMERATE)
        WIN.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        rk4(constraints_scene, add_forces)

        # drawing
        for i in scene:
            i.draw(WIN, trail=False)

        # pendulum connection line
        for i in range(len(objects_p2)):
            pygame.draw.line(WIN, colors[i], coords_to_pygame((p1.x, p1.y)), coords_to_pygame((objects_p2[i].x, objects_p2[i].y)), 1)
            pygame.draw.line(WIN, colors[i], coords_to_pygame((objects_p2[i].x, objects_p2[i].y)), coords_to_pygame((objects_p3[i].x, objects_p3[i].y)), 1)

        # total energy in scene
        total_energy = gravity.potential_energy()
        for i in scene:
            total_energy += i.energy()

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
