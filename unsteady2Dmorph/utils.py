import numpy as np


def Read_Two_Column_File(file_name, skiprows):
    with open(file_name, "r") as data:
        x = []
        y = []
        for i, line in enumerate(data):
            if i > skiprows:
                p = line.split()
                x.append(float(p[0]))
                y.append(float(p[1]))

    x = np.array(x)
    y = np.array(y)
    return x, y


def flip_panels(x_int, z_int):
    x_int_copy = np.copy(x_int)
    z_int_copy = np.copy(z_int)
    # Number of panels
    numPan = len(x_int_copy) - 1  # Number of panels

    # Check for direction of points

    edge = np.zeros(numPan)  # Initialize edge check value
    for i in range(numPan):  # Loop over all panels
        # Compute edge value for each panel
        edge[i] = (x_int_copy[i + 1] - x_int_copy[i]) * (
            z_int_copy[i + 1] + z_int_copy[i]
        )

    # Sum all panel edge values
    sumEdge = np.sum(edge)

    # If panels are CCW, flip them (don't if CW)
    if sumEdge < 0:  # If sum is negative
        # Display message in console
        print("Points are counter-clockwise.  Flipping.\n")

        # Flip the boundary points arrays
        x_int_copy = np.flip(x_int_copy)
        z_int_copy = np.flip(z_int_copy)

    elif sumEdge > 0:  # If sum is positive
        # Do nothing, display message in consolve
        print("Points are clockwise.  Not flipping.\n")

    return (x_int_copy, z_int_copy)
