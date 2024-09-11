import math
import numpy as np
from scipy.interpolate import interp1d


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


def flip_panels(x_int, z_int, desired_orientation):
    # Handle exception for desired_orientation
    if desired_orientation not in ["CW", "CCW"]:
        raise ValueError(
            "desired_orientation must be either 'CW' or 'CCW'. Please check the input."
        )
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
    if (sumEdge < 0 and desired_orientation == "CW") or (
        sumEdge > 0 and desired_orientation == "CCW"
    ):  # If sum is negative
        x_int_copy = np.flip(x_int_copy)
        z_int_copy = np.flip(z_int_copy)
    return (x_int_copy, z_int_copy)


# =============================================================================
# CST-related functions
# written using nomenclature from:
# "Inverse Airfoil Design Utilizing CST Parameterization"
# Kevin A. Lane and David D. Marshall, 2010
# =============================================================================


def GetClassFunctionValue(psi, N1=0.5, N2=1.0):
    return psi**N1 * (1 - psi) ** N2


def GetBinomialCoefficient(i, N):
    return math.factorial(N) / (math.factorial(i) * math.factorial(N - i))


def GetComponentFunction(psi, i, N):
    return GetBinomialCoefficient(i, N) * psi**i * (1 - psi) ** (N - i)


def DetermineCurvatureCoefficients(x_locations, y_locations, delta_zeta, N_bern):

    # generate the D matrix
    D = np.zeros((x_locations.size, N_bern + 1))
    for i, x_loc in enumerate(x_locations):
        C_N1_N2 = GetClassFunctionValue(x_loc)
        for j in range(N_bern + 1):
            D[i, j] = C_N1_N2 * GetComponentFunction(x_loc, j, N_bern)

    # determine the curvature coefficients
    A = np.linalg.lstsq(D, y_locations - x_locations * delta_zeta, rcond=None)[0]

    return A


# =============================================================================
# geometry-related functions
# =============================================================================
def split_airfoil_or_Cp_x_coord(entry_vec, x_vec, desired_orientation, tol_numerics):
    # Handle exception for desired_orientation
    if desired_orientation not in ["CW", "CCW"]:
        raise ValueError(
            "desired_orientation must be either 'CW' or 'CCW'. Please check the input."
        )
    index_LE_upper = np.where(np.abs(x_vec) < tol_numerics)[0][0]
    if desired_orientation == "CCW":
        entry_vec_upper = entry_vec[: (index_LE_upper + 1)]
        entry_vec_lower = entry_vec[(index_LE_upper + 1) :]
    else:
        entry_vec_upper = entry_vec[(index_LE_upper + 1) :]
        entry_vec_lower = entry_vec[: (index_LE_upper + 1)]

    return [entry_vec_upper, entry_vec_lower]


def smoothen_airfoil(x_int, z_int):
    num_pan_nodes = len(x_int)

    dx = np.diff(x_int)
    dz = np.diff(z_int)
    s = np.cumsum(np.sqrt(dx**2 + dz**2))
    s = np.insert(s, 0, 0)

    # Generate the new parameter s1 for higher resolution sampling
    s1 = np.linspace(0, s[-1], num_pan_nodes)

    # Remove duplicates in s
    s, unique_indices = np.unique(s, return_index=True)
    x1 = x_int[unique_indices]
    z1 = z_int[unique_indices]

    # Perform linear interpolation
    interp_x = interp1d(s, x1, kind="cubic")
    interp_z = interp1d(s, z1, kind="cubic")

    # Interpolate the new x and y values at the high resolution points s1
    x_high_res = interp_x(s1)
    z_high_res = interp_z(s1)

    return x_high_res, z_high_res
