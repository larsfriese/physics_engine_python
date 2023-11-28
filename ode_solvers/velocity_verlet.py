
def velocity_verlet(particles, add_forces):
    """
    This function takes a list of objects and a function that adds forces to the objects.
    It then calculates the next step in the simulation using the velocity verlet method.

    :param objects: list of objects
    :param add_forces: function that adds forces to the objects
    :return: None
    """

    vel = []
    for particle in particles:
        vel.append(particle.velocity_verlet_1())

    add_forces()

    for obj in objects:
        index = objects.index(obj)
        obj.velocity_verlet_2(vel[index])
