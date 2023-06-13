def rk4(objects, add_forces):

    """
    This function takes a list of objects and a function that adds forces to the objects.
    It then calculates the next step in the simulation using the RK4 method.

    :param objects: list of objects
    :param add_forces: function that adds forces to the objects
    :return: None
    """

    original_coordinates = []
    original_velocities = []

    for obj in objects:
        original_coordinates.append([obj.x, obj.y])
        original_velocities.append([obj.x_vel, obj.y_vel])

    add_forces()

    # first step
    kx0 = []
    kx1 = []
    kv0 = []

    for obj in objects:
        index = objects.index(obj)
        kx0x, kv0x, kx1x, kx0y, kv0y, kx1y = obj.rk4_1()
        kx0.append([kx0x, kx0y])
        kx1.append([kx1x, kx1y])
        kv0.append([kv0x, kv0y])
        obj.x = original_coordinates[index][0] + kx0x / 2
        obj.y = original_coordinates[index][1] + kx0y / 2
        obj.x_vel = original_velocities[index][0] + kv0x / 2
        obj.y_vel = original_velocities[index][1] + kv0y / 2

    add_forces()

    # second step
    kx2 = []
    kv1 = []

    for obj in objects:
        index = objects.index(obj)
        kx2x, kv1x, kx2y, kv1y = obj.rk4_2(kv0[index][0], kv0[index][1], original_velocities[index])
        kx2.append([kx2x, kx2y])
        kv1.append([kv1x, kv1y])
        obj.x = original_coordinates[index][0] + kx1[index][0] / 2
        obj.y = original_coordinates[index][1] + kx1[index][1] / 2
        obj.x_vel = original_velocities[index][0] + kv1[index][0] / 2
        obj.y_vel = original_velocities[index][1] + kv1[index][1] / 2

    add_forces()

    # third step
    kx3 = []
    kv2 = []

    for obj in objects:
        index = objects.index(obj)
        kx3x, kv2x, kx3y, kv2y = obj.rk4_3(original_velocities[index])
        kx3.append([kx3x, kx3y])
        kv2.append([kv2x, kv2y])
        obj.x = original_coordinates[index][0] + kx2[index][0]
        obj.y = original_coordinates[index][1] + kx2[index][1]
        obj.x_vel = original_velocities[index][0] + kv2[index][0]
        obj.y_vel = original_velocities[index][1] + kv2[index][1]

    add_forces()

    # fourth step
    kv3 = []

    for obj in objects:
        kv3x, kv3y = obj.rk4_4()
        kv3.append([kv3x, kv3y])

    # rk4 final step
    for obj in objects:
        index = objects.index(obj)
        obj.rk4_final(
            [original_coordinates[index], original_velocities[index], kx0[index], kx1[index], kx2[index], kx3[index],
             kv0[index], kv1[index], kv2[index], kv3[index]])


def velocity_verlet(objects, add_forces):

    """
    This function takes a list of objects and a function that adds forces to the objects.
    It then calculates the next step in the simulation using the velocity verlet method.

    :param objects: list of objects
    :param add_forces: function that adds forces to the objects
    :return: None
    """

    vel = []
    for obj in objects:
        vel.append(obj.velocity_verlet_1())

    add_forces()

    for obj in objects:
        index = objects.index(obj)
        obj.velocity_verlet_2(vel[index])