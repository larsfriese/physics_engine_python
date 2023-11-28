import pygame
import time, os

from ode_solvers import rk4
from objects import Object, Spring, Spring_to_mouse, coords_to_pygame
from constraints import ConstraintManager
from ode_solvers.rk4 import rk4

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
TIMESTEP = 1 / FRAMERATE  # 1 / (FRAMERATE / 60)


def main():
    run = True
    clock = pygame.time.Clock()
    tickcounter = 0

    p1 = Object(0, 100, 10, BLUE, 10, TIMESTEP)
    p1.x_vel = 0
    p1.y_vel = 0

    p2 = Object(200, 200, 10, RED, 10, TIMESTEP)
    p2.y_vel = 0
    p2.y_vel = 0

    spring = Spring(p1, p2, length=30, k=5)

    scene = [p1, p2, spring]

    # remove all objects of type spring from scene
    scene_without_springs = [i for i in scene if not isinstance(i, Spring)]

    # particles/objects used for the constraints
    constraints_scene = [p1, p2]

    # import/create constraints
    constraint_manager = ConstraintManager(constraints_scene)

    # control time
    start_time = time.time()
    energy_diff = 0
    last_energy = p1.energy() + p2.energy() + spring.energy()

    def add_forces():

        for i in scene:
            # if object is of type spring
            if isinstance(i, Spring) or isinstance(i, Spring_to_mouse):
                # add the spring forces
                i.add_forces()

        # calculate the constraint forces to the regular forces
        # to satisfy the constraints
        constraint_manager.update(constraints_scene)
        constraint_manager.rail_constraint(p1, '40 * cos((1/40)*x)')
        constraint_manager.rail_constraint(p2, 'x')
        constraint_manager.add_forces()

    mouse_down = False

    while run:
        clock.tick(FRAMERATE)
        WIN.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            # if mouse is clicked
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_down = True

                # create a spring between the mouse and the closest particle
                mouse_pos = pygame.mouse.get_pos()
                closest_particle = min(scene_without_springs, key=lambda x: x.distance(coords_to_pygame(mouse_pos)))
                spring_mouse = Spring_to_mouse(closest_particle, 1, 20, mouse_pos[0], mouse_pos[1])
                scene.append(spring_mouse)

            if mouse_down and event.type == pygame.MOUSEMOTION:
                # move the spring to the mouse
                mouse_pos = pygame.mouse.get_pos()
                spring_mouse.mouse_x = mouse_pos[0]
                spring_mouse.mouse_y = mouse_pos[1]

            if event.type == pygame.MOUSEBUTTONUP:
                # remove all Spring_to_mouse objects from scene
                scene = [i for i in scene if not isinstance(i, Spring_to_mouse)]


        # update positions
        rk4([p1, p2], add_forces)
        #ode_solvers.velocity_verlet([p1, p2], add_forces)

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