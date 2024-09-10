import numpy as np
import matplotlib.pyplot as plt


def geometry(N, c, x_pitch, beta=0, x_flap=0):
    """
    assumes beta is in radians, and positive in the clockwise direction
    """

    def rotate_points(x_vals, z_vals, flap_position, beta):
        args_flap = np.where(x_vals >= flap_position)
        x_new = flap_position + (x_vals[args_flap] - flap_position) * np.cos(-beta)
        z_new = 0 + (x_vals[args_flap] - flap_position) * np.sin(-beta)
        x_vals_new = np.copy(x_vals)
        z_vals_new = np.copy(z_vals)
        x_vals_new[args_flap] = x_new
        z_vals_new[args_flap] = z_new
        return x_vals_new, z_vals_new

    # Initialize positions of vortex points
    x_vort = np.zeros(N)
    z_vort = np.zeros(N)

    # Initialize positions of control points
    x_col = np.zeros(N)
    z_col = np.zeros(N)

    # Initialize the components of the normal and tangential vector of the panel
    nx = np.zeros(N)
    nz = np.zeros(N)
    tx = np.zeros(N)
    tz = np.zeros(N)

    # initialize the angles of the panels with respect to the x-axis
    alpha = np.zeros(N)

    # Panel length
    l_panel = np.zeros(N)

    # Initialize the x-locations of the
    x_int = np.linspace(-x_pitch, c - x_pitch, N + 1)
    z_int = np.zeros(N + 1)

    # if using a flap, make sure that a panel node is at the hinge position
    # (to avoid errors in the geometry of the flap)
    if x_flap >= 0:
        flap_position = x_flap - x_pitch
        args_not_flap_int = np.where(x_int < flap_position)
        if args_not_flap_int[0].size > 0:
            args_closest_left_int = args_not_flap_int[0][-1]
            if np.abs(x_int[args_closest_left_int] - flap_position) < np.abs(
                x_int[args_closest_left_int + 1] - flap_position
            ):
                x_int[args_closest_left_int] = flap_position
            else:
                x_int[args_closest_left_int + 1] = flap_position
    for i in range(N):
        l_panel[i] = x_int[i + 1] - x_int[i]

    for i in range(N):
        x_vort[i] = 0.25 * (x_int[i + 1] - x_int[i]) + x_int[i]
        x_col[i] = 0.75 * (x_int[i + 1] - x_int[i]) + x_int[i]

    # Define coefficient matrix for self-induced velocity
    gamma = 1
    coefficient_matrix = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            vx, vz = VOR2D(gamma, x_col[i], z_col[i], x_vort[j], z_vort[j])
            coefficient_matrix[i, j] = vx + vz

    if x_flap >= 0:
        [x_int, z_int] = rotate_points(x_int, z_int, flap_position, beta)
        [x_col, z_col] = rotate_points(x_col, z_col, flap_position, beta)
        [x_vort, z_vort] = rotate_points(x_vort, z_vort, flap_position, beta)
    for i in range(N):
        nx[i] = np.sin(z_vort[i] / (x_int[i + 1] - x_int[i]))
        nz[i] = np.cos(z_vort[i] / (x_int[i + 1] - x_int[i]))
        alpha[i] = np.arctan((z_int[i] - z_int[i + 1]) / (x_int[i + 1] - x_int[i]))
        nx[i] = np.sin(alpha[i])
        nz[i] = np.cos(alpha[i])
        tx[i] = np.cos(alpha[i])
        tz[i] = -np.sin(alpha[i])

    x_TE = x_int[-1]
    z_TE = z_int[-1]

    if x_flap >= 0:
        col_indices_flap = np.where(x_col >= flap_position)[0]
    else:
        col_indices_flap = []

    return (
        x_vort,
        x_col,
        z_vort,
        z_col,
        tx,
        tz,
        nx,
        nz,
        coefficient_matrix,
        l_panel,
        alpha,
        x_TE,
        z_TE,
        col_indices_flap,
    )


def geometry_airfoil(
    airfoil_name, flap_position, theta=0, x_pitch=0, z_pitch=0, skiprows=0
):
    """
    Function to generate the geometry of an airfoil using the NACA four-digit
    series. The airfoil is discretized into a number of panels, and the
    coordinates of the panel nodes, control points, and panel angles are
    calculated.

    Inputs:
    - airfoil_name: name of the airfoil file
    - flap_position: position of the flap hinge
    - theta: angle of rotation of the airfoil [rad]
    - x_pitch: x-coordinate of the pitching axis
    - z_pitch: z-coordinate of the pitching axis
    - skiprows: number of rows to skip in the airfoil file

    Outputs:
    - x_int: x-coordinates of the panel nodes
    - z_int: z-coordinates of the panel nodes
    - xc: x-coordinates of the panel control points
    - zc: z-coordinates of the panel control points
    - numPan: number of panels
    - s: panel lengths
    - nx: x-components of the panel normal vectors
    - nz: z-components of the panel normal vectors
    - tx: x-components of the panel tangent vectors
    - tz: z-components of the panel tangent vectors
    """

    # Define the NACA reference number and number of panels
    mpt = "0012"  # NACA reference number
    n = 400  # Number of panels, must be even

    # Set non-dimensional x-coordinates of the airfoil
    x_int = 0.5 * (
        1 - np.cos(np.linspace(-np.pi, np.pi, n + 1))
    )  # Nonlinear distribution

    # Calculate non-dimensional y-coordinates of the airfoil
    z_int = nacafourdigit(x_int, int(n / 2), mpt, 1)

    # x_int, z_int = Read_Two_Column_File(airfoil_name, skiprows)

    # flip the panels if needed
    # (x_int, z_int) = flip_panels(x_int, z_int)

    # Shift the reference system to the pitching axis
    x_int -= x_pitch
    z_int -= z_pitch

    # if using a flap, make sure that a panel node is at the hinge position
    # (to avoid errors in the geometry of the flap)
    x_flap = flap_position[0]

    R_theta = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )
    xzrot = np.dot(R_theta, np.array([x_int, z_int]))

    x_int = xzrot[0, :]
    z_int = xzrot[1, :]
    x_int_before = np.copy(x_int)
    z_int_before = np.copy(z_int)

    # Calculate panel angles
    theta = np.arctan2(z_int[1:] - z_int[:-1], x_int[1:] - x_int[:-1])

    # Calculate control points
    xc = x_int[:-1] + (x_int[1:] - x_int[:-1]) / 2
    zc = z_int[:-1] + (z_int[1:] - z_int[:-1]) / 2

    x_int_before = np.copy(x_int)
    z_int_before = np.copy(z_int)

    # if x_flap >= 0:
    #     x_flap -= x_pitch
    #     x_upper = x_int[z_int >= 0]
    #     x_lower = x_int[z_int < 0]
    #     z_upper = z_int[z_int >= 0]
    #     z_lower = z_int[z_int < 0]
    #     theta_upper = theta[zc >= 0]
    #     theta_lower = theta[zc < 0]
    #     args_not_flap_int_upper = np.where(x_upper < x_flap)
    #     args_not_flap_int_lower = np.where(x_lower < x_flap)
    #     if args_not_flap_int_upper[0].size > 0:
    #         args_closest_left_int_upper = args_not_flap_int_upper[0][-1]
    #         point_upper_left = np.array(
    #             [
    #                 x_upper[args_closest_left_int_upper],
    #                 z_upper[args_closest_left_int_upper],
    #             ]
    #         )
    #         point_upper_right = np.array(
    #             [
    #                 x_upper[args_closest_left_int_upper + 1],
    #                 z_upper[args_closest_left_int_upper + 1],
    #             ]
    #         )
    #         if np.linalg.norm(
    #             point_upper_left - flap_position
    #             < np.linalg.norm(point_upper_right - flap_position)
    #         ):
    #             index_closest_upper = args_closest_left_int_upper
    #         else:
    #             print("goes where it should upper")
    #             index_closest_upper = args_closest_left_int_upper + 1
    #         x_upper[index_closest_upper] = x_flap
    #         z_upper[index_closest_upper] = z_upper[index_closest_upper - 1] + np.abs(
    #             x_flap - x_upper[index_closest_upper - 1]
    #         ) * np.tan(theta_upper[index_closest_upper - 1])

    #     if args_not_flap_int_lower[0].size > 0:
    #         args_closest_left_int_lower = args_not_flap_int_lower[0][0]
    #         point_lower_left = np.array(
    #             [
    #                 x_lower[args_closest_left_int_lower],
    #                 z_lower[args_closest_left_int_lower],
    #             ]
    #         )
    #         point_lower_right = np.array(
    #             [
    #                 x_lower[args_closest_left_int_lower - 1],
    #                 z_lower[args_closest_left_int_lower - 1],
    #             ]
    #         )
    #         if np.linalg.norm(point_lower_left - flap_position) < np.linalg.norm(
    #             point_lower_right - flap_position
    #         ):
    #             index_closest_lower = args_closest_left_int_lower
    #         else:
    #             print("goes where it should lower")
    #             index_closest_lower = args_closest_left_int_lower - 1
    #         x_lower[index_closest_lower] = x_flap
    #         z_lower[index_closest_lower] = z_lower[index_closest_lower - 1] + (
    #             x_flap - x_lower[index_closest_lower - 1]
    #         ) * np.tan(theta_lower[index_closest_lower - 1])
    #         print(
    #             "addition to lower:",
    #             np.abs(x_flap - x_lower[index_closest_lower - 1])
    #             * np.tan(theta_lower[index_closest_lower - 1]),
    #         )
    #     x_int = np.concatenate([x_lower, x_upper])
    #     z_int = np.concatenate([z_lower, z_upper])

    #     plt.figure(1)
    #     plt.plot(x_flap, flap_position[1], "o", label="hinge")
    #     plt.axvline(x=x_flap, color="b", label="axvline - full height")
    #     plt.plot(x_int_before, z_int_before, "*", label="before")
    #     # plt.plot(x_int, z_int, "*", label="after")
    #     # plt.plot(x_lower, z_lower, "o", label="lower")
    #     # plt.plot(point_lower_left[0], point_lower_left[1], "o", label="lower left")
    #     # plt.plot(point_lower_right[0], point_lower_right[1], "o", label="lower right")
    #     # plt.plot(point_upper_left[0], point_upper_left[1], "o", label="upper left")
    #     # plt.plot(point_upper_right[0], point_upper_right[1], "o", label="upper right")
    #     # plt.plot(
    #     #     x_int_before[index_closest_lower],
    #     #     z_int_before[index_closest_lower],
    #     #     "o",
    #     #     label="chosen lower",
    #     # )
    #     # plt.plot(x_int, z_int, "*", label="after")
    #     plt.legend()
    #     plt.show()

    numPan = len(x_int) - 1  # Number of panels

    # Calculate control points
    xc = x_int[:-1] + (x_int[1:] - x_int[:-1]) / 2
    zc = z_int[:-1] + (z_int[1:] - z_int[:-1]) / 2

    # Calculate panel lengths
    s = np.sqrt((x_int[1:] - x_int[:-1]) ** 2 + (z_int[1:] - z_int[:-1]) ** 2)
    # Calculate tangent vectors
    tx = (x_int[1:] - x_int[:-1]) / s
    tz = (z_int[1:] - z_int[:-1]) / s
    # Calculate normal vectors
    nx = -tz
    nz = tx

    # # Calculate linearly varying vortex influence coefficient matrices
    # [Au, Av] = vpminf(xc, zc, xc, zc, tx, tz, nx, nz, s)

    # # Calculate normal velocity influence coefficients
    # An = Au * nx[:, np.newaxis] + Av * nz[:, np.newaxis]

    # # Calculate tangential velocity influence coefficients
    # At = Au * tx[:, np.newaxis] + Av * tz[:, np.newaxis]

    # # Add Kutta condition
    # A_Kutta = np.zeros(numPan + 1)
    # A_Kutta = At[0, :] + At[-1, :]

    # coeff_matrix = np.vstack((An, A_Kutta))

    return (x_int, z_int, xc, zc, numPan, s, nx, nz, tx, tz)


def compute_static_polar_alpha_sweep(N, c, x_theta, alpha_vals, Uinf, beta=0, x_beta=0):
    Cl_vals = np.zeros(alpha_vals.size)
    (
        x_vort,
        x_col,
        z_vort,
        z_col,
        tx,
        tz,
        nx,
        nz,
        coefficient_matrix,
        l_panel,
        alpha,
        x_TE,
        z_TE,
        col_indices_flap,
    ) = geometry(N, c, x_theta, beta, x_beta)
    for i in range(alpha_vals.size):
        alpha = alpha_vals[i]
        RHS = np.zeros(N)
        for j in range(N):
            RHS[j] = -Uinf * (np.cos(alpha) * nx[j] + np.sin(alpha) * nz[j])
        # Solve the matrix equation for the circulation gamma
        gamma = np.linalg.solve(coefficient_matrix, RHS)

        Cl_vals[i] = Uinf * np.sum(gamma) / (1 / 2 * Uinf**2 * c)

    return Cl_vals


def compute_static_polar_beta_sweep(N, c, x_theta, beta_vals, Uinf, alpha_0, x_beta=0):
    Cl_vals = np.zeros(beta_vals.size)
    for i in range(beta_vals.size):
        (
            x_vort,
            x_col,
            z_vort,
            z_col,
            tx,
            tz,
            nx,
            nz,
            coefficient_matrix,
            l_panel,
            alpha,
            x_TE,
            z_TE,
            col_indices_flap,
        ) = geometry(N, c, x_theta, beta_vals[i], x_beta)
        RHS = np.zeros(N)
        for j in range(N):
            RHS[j] = -Uinf * (np.cos(alpha_0) * nx[j] + np.sin(alpha_0) * nz[j])
        # Solve the matrix equation for the circulation gamma
        gamma = np.linalg.solve(coefficient_matrix, RHS)

        Cl_vals[i] = Uinf * np.sum(gamma) / (1 / 2 * Uinf**2 * c)

    return Cl_vals


def newtonVPM(newtonVPM_dict):
    """
    Written by Grigorios Dimitriadis, May 2023.
    This code accompanies the book Unsteady Aerodynamics - Potential and
    Vortex Methods, G. Dimitriadis, Wiley, 2023. If any part of the code is
    reused, the author and book must be acknowledged.

    Calculates Newton-Raphson function for unsteady vortex panel method with linearly varying strength.

    Parameters are based on the MATLAB function. Outputs are:
    Fsv, xwp, ywp, xcwp, ycwp, nxwp, nywp, tauxwp, tauywp, gammawp, gamma, uwp, wwp, utang
    """

    # Read the data from the dictionary
    lwp = newtonVPM_dict["lwp"]
    thetawp = newtonVPM_dict["thetawp"]
    x = newtonVPM_dict["x"]
    z = newtonVPM_dict["z"]
    xc = newtonVPM_dict["xc"]
    zc = newtonVPM_dict["zc"]
    xw = newtonVPM_dict["xw"]
    zw = newtonVPM_dict["zw"]
    Gammaw = newtonVPM_dict["Gammaw"]
    nx = newtonVPM_dict["nx"]
    nz = newtonVPM_dict["nz"]
    taux = newtonVPM_dict["tx"]
    tauz = newtonVPM_dict["tz"]
    s = newtonVPM_dict["s"]
    U = newtonVPM_dict["U"]
    dt = newtonVPM_dict["dt"]
    uw_wp = newtonVPM_dict["uw_wp"]
    ww_wp = newtonVPM_dict["ww_wp"]
    An = newtonVPM_dict["An"]
    Atau = newtonVPM_dict["Atau"]
    bvec = newtonVPM_dict["bvec"]
    btau = newtonVPM_dict["btau"]
    utauw = newtonVPM_dict["utauw"]
    vnw = newtonVPM_dict["vnw"]
    Gamma_old = newtonVPM_dict["Gamma_old"]
    it = newtonVPM_dict["it"]

    n = len(xc)  # Number of panels on airfoil

    # Wake panel coordinates
    xwp = (x[0] + x[n]) / 2 + np.array([0, lwp * np.cos(thetawp)])
    zwp = (z[0] + z[n]) / 2 + np.array([0, lwp * np.sin(thetawp)])

    # Calculate collocation point of wake panel
    xcwp = (xwp[0] + xwp[1]) / 2
    zcwp = (zwp[0] + zwp[1]) / 2

    # Calculate unit tangent vectors on wake panel
    tauxwp = np.cos(thetawp)
    tauzwp = np.sin(thetawp)

    # Calculate unit normal vectors on wake panel
    nxwp = -np.sin(thetawp)
    nzwp = np.cos(thetawp)

    # Influence coefficient of wake panel on airfoil panels
    Auwp, Awwp = svpminf(xc, zc, xcwp, zcwp, tauxwp, tauzwp, nxwp, nzwp, lwp)

    # Normal wake panel influence coefficient
    Anwp = np.array(Auwp[:, 0]) * np.array(nx) + np.array(Awwp[:, 0]) * np.array(nz)

    # Tangential wake panel influence coefficient
    Atauwp = np.array(Auwp[:, 0]) * taux + np.array(Awwp[:, 0]) * tauz

    # Equal pressure at the trailing edge Kutta condition
    s_bar = (np.concatenate([[0], s]) + np.concatenate([s, [0]])) / 2

    s_bar_star = s_bar[:n]
    bar_An = An - (1 / lwp) * np.outer(Anwp, s_bar)
    bar_a0n = bvec - vnw - Anwp * Gamma_old / lwp
    bar_An_star = bar_An[:, :n]
    bar_an_star = bar_An[:, n]
    bgamma = -np.linalg.solve(bar_An_star, bar_an_star)
    b0 = np.linalg.solve(bar_An_star, bar_a0n)
    bar_Atau = Atau - (1 / lwp) * np.outer(Atauwp, s_bar)
    bar_a0tau = Atauwp * Gamma_old / lwp + utauw
    bar_Atau_star = bar_Atau[:, :n]
    bar_atau_star = bar_Atau[:, n]
    cgamma = np.dot(bar_Atau_star, bgamma) + bar_atau_star
    c0 = np.dot(bar_Atau_star, b0) + bar_a0tau
    dgamma = np.dot(s_bar_star, bgamma) + s[-1] / 2
    d0 = np.dot(s_bar_star, b0)

    if it == 0:
        a2 = cgamma[n - 1] ** 2 - cgamma[0] ** 2
        a1 = 2 * (
            cgamma[n - 1] * (c0[n - 1] + btau[n - 1]) - cgamma[0] * (c0[0] + btau[0])
        )
        a0 = (c0[n - 1] + btau[n - 1]) ** 2 - (c0[0] + btau[0]) ** 2
    else:
        a2 = cgamma[n - 1] ** 2 - cgamma[0] ** 2
        a1 = 2 * (
            cgamma[n - 1] * (c0[n - 1] + btau[n - 1])
            - cgamma[0] * (c0[0] + btau[0])
            + dgamma / dt
        )
        a0 = (
            (c0[n - 1] + btau[n - 1]) ** 2
            - (c0[0] + btau[0]) ** 2
            + 2 * (d0 - Gamma_old) / dt
        )

    Kutta_poly = np.array([a2, a1, a0])

    discr = Kutta_poly[1] ** 2 - 4 * Kutta_poly[0] * Kutta_poly[2]  # Discriminant

    gamma_np1 = (-Kutta_poly[1] + np.sqrt(discr)) / (2 * Kutta_poly[0])  # Solution
    gamma_star = np.dot(bgamma, gamma_np1) + b0
    gamma = np.concatenate([gamma_star, [gamma_np1]])
    gammawp = -(1 / lwp) * np.dot(s_bar, gamma) + Gamma_old / lwp

    # Calculate total tangent velocities on control points
    utang = np.dot(Atau, gamma) + Atauwp * gammawp + utauw

    # Calculate airfoil panel influence coefficients on wake control point
    Au, Av = vpminf(xcwp, zcwp, xc, zc, taux, tauz, nx, nz, s)

    if it > 0:
        # Calculate influence coefficients of point wake vortices on wake control point
        Buwp, Bwwp = lvminf(xcwp, zcwp, xw[:it], zw[:it])
        uw_wp = np.dot(Buwp, Gammaw[:it])  # Horizontal velocity
        ww_wp = np.dot(Bwwp, Gammaw[:it])  # Vertical velocity

    # Total velocity at wake control point
    uwp = np.dot(Au, gamma) + uw_wp + U
    wwp = np.dot(Av, gamma) + ww_wp

    # New wake panel angle
    thetawp_new = np.arctan2(wwp, uwp)

    # New wake panel length
    lwp_new = np.sqrt(uwp**2 + wwp**2) * dt

    # Convergence criterion
    Fsv = np.array([lwp - lwp_new, thetawp - thetawp_new])

    output_dictionary = {
        "Fsv": Fsv,
        "xwp": xwp,
        "zwp": zwp,
        "xcwp": xcwp,
        "zcwp": zcwp,
        "nxwp": nxwp,
        "nzwp": nzwp,
        "tauxwp": tauxwp,
        "tauzwp": tauzwp,
        "gammawp": gammawp,
        "gamma": gamma,
        "uwp": uwp,
        "wwp": wwp,
        "utang": utang,
    }

    return output_dictionary


def lvminf(xc, yc, xb, yb):
    """
    Written by Grigorios Dimitriadis, May 2023.
    This code accompanies the book Unsteady Aerodynamics - Potential and
    Vortex Methods, G. Dimitriadis, Wiley, 2023. If any part of the code is
    reused, the author and book must be acknowledged.

    Calculates the influence coefficients induced by lumped vortices lying at xb, yb
    at control points xc, yc using the standard vortex model.

    Parameters:
    - xc: 1D array of control points x-coordinates
    - yc: 1D array of control points y-coordinates
    - xb: 1D array of lumped vortex position x-coordinates
    - yb: 1D array of lumped vortex position y-coordinates

    Returns:
    - Au: 2D array of horizontal velocity influence coefficients
    - Av: 2D array of vertical velocity influence coefficients
    """
    # Transform the influenced coordinates to arrays (useful if only one panel
    # is influenced)
    if isinstance(xc, (np.floating, float)):
        xc = np.array([xc])
        yc = np.array([yc])

    nc = len(xc)  # Number of control points
    nb = len(xb)  # Number of lumped vortices

    # Initialize horizontal and vertical velocity influence coefficient matrices
    Au = np.zeros((nc, nb))
    Av = np.zeros((nc, nb))

    # Cycle through control points
    for i in range(nc):
        # Cycle through lumped vortices
        for j in range(nb):
            # Square of distance between current vortex and current control point
            rsquared = (xc[i] - xb[j]) ** 2 + (yc[i] - yb[j]) ** 2

            if np.sqrt(rsquared) < 1e-12:
                # If the distance is small, set both influences to zero
                Au[i, j] = 0
                Av[i, j] = 0
            else:
                # Calculate horizontal influence coefficient (Equation 4.20)
                Au[i, j] = 1 / (2 * np.pi) * (yc[i] - yb[j]) / rsquared

                # Calculate vertical influence coefficient (Equation 4.21)
                Av[i, j] = -1 / (2 * np.pi) * (xc[i] - xb[j]) / rsquared

    return Au, Av


def compute_stagnation_phi(
    x_airfoil,
    z_airfoil,
    s,
    stagnation_phi_choice,
    aero_model,
    N_chords,
    xc,
    zc,
    tx,
    tz,
    nx,
    nz,
    c,
    gamma_bound,
    N_timestep,
    xc_wp,
    zc_wp,
    tx_wp,
    tz_wp,
    nx_wp,
    nz_wp,
    l_wp,
    gamma_wp,
    xw,
    zw,
    gamma_w,
):
    """ """

    # Compute the location of the stagnation point, and the associated panel
    # length
    N = int(len(x_airfoil) - 1)
    if stagnation_phi_choice == "LE":
        index_panel_stagnation = int(N / 2) - 1
        ds = s[int(N / 2) - 1]
        x_LE = x_airfoil[int(N / 2)]
    else:
        # Find all the intersection between the airfoil panels and the x axis
        indexes_x_axis_intersect = []
        for i in range(len(z_airfoil[:-1])):
            if z_airfoil[i] * z_airfoil[i + 1] < 0:
                indexes_x_axis_intersect.append(i)

        # Find the intersection point that is closest to the LE
        x_locations_intersections = [
            x_airfoil[index] for index in indexes_x_axis_intersect
        ]
        closest_index = np.argmin(
            np.abs(x_locations_intersections - x_airfoil[int(N / 2 - 1)])
        )
        closest_index = indexes_x_axis_intersect[closest_index]

        # Associate the panel node to the stagnation point
        x_intersect = x_airfoil[closest_index] - z_airfoil[closest_index] * (
            x_airfoil[closest_index + 1] - x_airfoil[closest_index]
        ) / (z_airfoil[closest_index + 1] - z_airfoil[closest_index])
        distance_i = np.sqrt(
            (x_intersect - x_airfoil[closest_index]) ** 2
            + z_airfoil[closest_index] ** 2
        )
        distance_i_p1 = np.sqrt(
            (x_intersect - x_airfoil[closest_index + 1]) ** 2
            + z_airfoil[closest_index + 1] ** 2
        )
        if distance_i < distance_i_p1:
            x_LE = x_airfoil[closest_index]
            index_panel_stagnation = closest_index - 1
        else:
            x_LE = x_airfoil[closest_index + 1]
            index_panel_stagnation = closest_index

        # Associate a panel length to the stagnation point

        ds = s[closest_index]

    # Nonlinear spacing for the panels, first panel same length as
    # leading edge panel
    angles_spanels = np.arange(0, np.pi / 2, np.arccos(1 - ds / (c * N_chords)))
    spanels = (1 - np.cos(angles_spanels)) * N_chords * c

    # x and z coordinates of the panels
    x_pan = x_LE - np.flip(spanels)
    z_pan = np.zeros(len(x_pan))

    # Panel control points
    xc_pan = (x_pan[1:] + x_pan[:-1]) / 2
    zc_pan = (z_pan[1:] + z_pan[:-1]) / 2

    # Calculate panel lengths
    s_pan = np.sqrt((x_pan[1:] - x_pan[:-1]) ** 2 + (z_pan[1:] - z_pan[:-1]) ** 2)

    # Calculate tangent vectors on upstream panels
    taux_pan = (x_pan[1:] - x_pan[:-1]) / s_pan
    tauz_pan = (z_pan[1:] - z_pan[:-1]) / s_pan

    # Calculate the velocities induces onto the potential panels

    # Calculate influence coefficients on panel control points
    # Influence of airfoil panels
    Au_phi, Av_phi = vpminf(xc_pan, zc_pan, xc, zc, tx, tz, nx, nz, s)
    u_pan = np.dot(Au_phi, gamma_bound)
    w_pan = np.dot(Av_phi, gamma_bound)

    # If using unsteady aerodynamic model, add the effect of the wake
    if aero_model in ["quasi-steady", "unsteady"]:
        # Influence of wake panel
        Au_phi_wp, Av_phi_wp = svpminf(
            xc_pan, zc_pan, xc_wp, zc_wp, tx_wp, tz_wp, nx_wp, nz_wp, l_wp
        )
        Au_phi_wp = Au_phi_wp.flatten()
        Av_phi_wp = Av_phi_wp.flatten()

        # Influence of point wake vortices
        B_u_phi, B_v_phi = lvminf(xc_pan, zc_pan, xw[0:N_timestep], zw[0:N_timestep])

        u_pan += Au_phi_wp * gamma_wp + np.dot(B_u_phi, gamma_w[:N_timestep])
        w_pan += Av_phi_wp * gamma_wp + np.dot(B_v_phi, gamma_w[:N_timestep])
    # Tangential velocity on panel
    utau_pan = u_pan * taux_pan + w_pan * tauz_pan

    # Potential at the LE
    phi_LE = np.sum(utau_pan * s_pan)

    #

    return index_panel_stagnation, phi_LE


# Helper functions
# determining X and Y terms for recursive marching formula for approximation of Duhamel's integral
def time2semichord(time, Uinf, chord):
    return 2 * Uinf * time / chord


def semichord2time(s, Uinf, chord):
    return s / 2 / Uinf * chord


def duhamel_approx(
    Xi, Yi, delta_s, delta_alpha, order=2, A1=0.3, A2=0.7, b1=0.14, b2=0.53
):
    A1 = 0.165
    A2 = 0.335
    b1 = 0.0455
    b2 = 0.3
    # determine the next values of X and Y, named Xip1 and Yip1
    if order == 1:
        Xip1 = Xi * np.exp(-b1 * delta_s) + A1 * delta_alpha
        Yip1 = Yi * np.exp(-b2 * delta_s) + A2 * delta_alpha
    elif order == 2:
        Xip1 = Xi * np.exp(-b1 * delta_s) + A1 * delta_alpha * np.exp(-b1 * delta_s / 2)
        Yip1 = Yi * np.exp(-b2 * delta_s) + A2 * delta_alpha * np.exp(-b2 * delta_s / 2)
    else:
        Xip1 = Xi * np.exp(-b1 * delta_s) + A1 * delta_alpha * (
            (1 + 4 * np.exp(-b1 * delta_s / 2) + np.exp(-b1 * delta_s)) / 6
        )
        Yip1 = Yi * np.exp(-b2 * delta_s) + A2 * delta_alpha * (
            (1 + 4 * np.exp(-b2 * delta_s / 2) + np.exp(-b2 * delta_s)) / 6
        )

    return Xip1, Yip1


# define function for circulatory force, potential flow
def circulatory_normal_force(dCn_dalpha, alpha_equivalent, alpha0):
    return dCn_dalpha * (alpha_equivalent - alpha0)


# deficiency function for non-circulatory normal force
def deficiency_function(
    Dnoncirc_i, delta_dalpha_dt, delta_t, chord, asound=343, kalpha=0.75
):
    # a sound is the speed of sound
    TI = chord / asound
    Dnoncirc_ip1 = Dnoncirc_i * np.exp(
        -delta_t / (kalpha * TI)
    ) + delta_dalpha_dt * np.exp(-delta_t / (2 * kalpha * TI))
    return Dnoncirc_ip1


def non_circulatory_normal_force(dalpha_dt, chord, Uinf, Dnoncirc, kalpha=0.75):
    return 4 * kalpha * chord / Uinf * (dalpha_dt - Dnoncirc)


def get_Theodorsen_approx(
    force_type,
    alpha,
    t_array,
    alpha_t0,
    amplitude_alpha,
    omega,
    Uinf,
    chord,
    alpha0=0,
    dCn_dalpha=2 * np.pi,
):
    """
    Only used for pitching
    Assumes constant timestep
    """
    alpha = amplitude_alpha * np.sin(omega * t_array) + alpha_t0
    dt = t_array[1] - t_array[0]
    dalpha_dt = np.gradient(alpha, t_array)  # calculate the time derivative of alpha
    sarray = time2semichord(t_array, Uinf, chord)
    alphaqs = alpha + dalpha_dt * (chord / 2) / Uinf
    dalphaqs_dt = np.gradient(
        alphaqs, t_array
    )  # calculate the time derivative of the quasi-steady alpha
    # define arrays for X,Y
    Xarray = np.zeros(np.shape(t_array))
    Yarray = np.zeros(np.shape(t_array))
    # define the array of alpha_equivalent
    alpha_equivalent = np.zeros(np.shape(t_array))
    alpha_equivalent[0] = alphaqs[0]
    # march solution in time for alpha_E
    for i, val in enumerate(t_array[:-1]):
        Xarray[i + 1], Yarray[i + 1] = duhamel_approx(
            Xarray[i], Yarray[i], sarray[i + 1] - sarray[i], alphaqs[i + 1] - alphaqs[i]
        )
    alpha_equivalent = alphaqs - Xarray - Yarray
    Cnormal_circ = circulatory_normal_force(dCn_dalpha, alpha_equivalent, alpha0)
    # define arrays for Dnoncirc, the deficiency function for non-circulatory loading
    Dnoncirc = np.zeros(np.shape(t_array))
    # march solution in time
    for i, val in enumerate(t_array[:-1]):
        Dnoncirc[i + 1] = deficiency_function(
            Dnoncirc[i], dalphaqs_dt[i + 1] - dalphaqs_dt[i], dt, chord
        )

    # Cnormal_circ = circulatory_normal_force(dCn_dalpha, alpha_equivalent, alpha0)
    Cnormal_noncirc = non_circulatory_normal_force(dalphaqs_dt, chord, Uinf, Dnoncirc)
    Cnormal = Cnormal_noncirc + Cnormal_circ
    if force_type == "circulatory":
        return alpha_equivalent, Cnormal_circ
    elif force_type == "non-circulatory":
        return alpha_equivalent, Cnormal_noncirc
    elif force_type == "combined":
        return alpha_equivalent, Cnormal


def nacafourdigit(xp, m, mpt, teclosed):
    """
    This code accompanies the book Unsteady Aerodynamics - Potential and
    Vortex Methods, G. Dimitriadis, Wiley, 2023

    Function that gives the stations and ordinates of any NACA four-digit airfoil

    Parameters:
    xp: x-coordinates starting from 1, going to 0 and back up to 1.
        Length of xp must be 2*m+1.
    mpt: 4-digit serial number of airfoil, e.g. 2412 or 0008.
    teclosed: if teclosed=0, the normal NACA four digit thickness equation is used.
              if teclosed=1, a modified equation is used to close the trailing edge.

    Returns:
    zp: z-coordinates of the complete airfoil shape. They are arranged from
        the lower trailing edge towards the leading edge and then towards the
        upper trailing edge.
    zpcamb: z-coordinates of the camber line only. They are arranged from the
            leading edge to the trailing edge.
    """
    first_digit = int(mpt[0])
    second_digit = int(mpt[1])
    last_two_digits = int(mpt[2:])

    # Calculate airfoil parameters from serial number
    maxcamb = np.floor(first_digit) / 100  # Maximum camber
    maxpos = np.floor(second_digit) / 10  # Position of maximum camber
    t = last_two_digits / 100  # Thickness ratio

    xpup = xp[m:]  # Upper surface x-coordinates

    if teclosed == 0:
        # Calculate upper surface y coordinates from equation in Abbott and Von Doenhoff
        zpthick = (
            t
            / 0.2
            * (
                0.2969 * np.sqrt(xpup)
                - 0.126 * xpup
                - 0.3516 * xpup**2
                + 0.2843 * xpup**3
                - 0.1015 * xpup**4
            )
        )
    elif teclosed == 1:
        # Adapt highest order coefficient to close the trailing edge
        zpthick = (
            t
            / 0.2
            * (
                0.2969 * np.sqrt(xpup)
                - 0.126 * xpup
                - 0.3516 * xpup**2
                + 0.2843 * xpup**3
                - 0.1036 * xpup**4
            )
        )
    else:
        raise ValueError("teclosed must be equal to 0 or 1")

    # Calculate camber line
    zpcamb = np.zeros(m + 1)
    if maxpos != 0:
        iko = np.where(xpup <= maxpos)[0]
        zpcamb[iko] = maxcamb / maxpos**2 * (2 * maxpos * xpup[iko] - xpup[iko] ** 2)
        iko = np.where(xpup > maxpos)[0]
        zpcamb[iko] = (
            maxcamb
            / ((1 - maxpos) ** 2)
            * ((1 - 2 * maxpos) + 2 * maxpos * xpup[iko] - xpup[iko] ** 2)
        )

    # Assemble complete z-coordinates from lower trailing edge to upper trailing edge
    zp = np.concatenate([np.flip(-zpthick + zpcamb), zpthick[1:] + zpcamb[1:]])

    return zp


# Determine the velocity at (x,z) due to a vortex element at (xj,zj)
def VOR2D(gamma, x, z, xj, zj, epsilon=1e-6):
    r = np.sqrt((x - xj) ** 2 + (z - zj) ** 2)
    if r < epsilon:
        u = w = 0
    else:
        u = gamma / (2 * np.pi * r**2) * (z - zj)
        w = -gamma / (2 * np.pi * r**2) * (x - xj)
    return u, w


def svpminf(xci, zci, xcj, zcj, tauxj, tauzj, nxj, nzj, sj):
    """
    This code accompanies the book Unsteady Aerodynamics - Potential and
    Vortex Methods, G. Dimitriadis, Wiley, 2023. If any part of the code is
    reused, the author and book must be acknowledged.

    Influence coefficient matrices for the source and vortex panel method with constant strength.
    Calculates the horizontal and vertical velocity influence of source and vortex panel j on point i.

    Parameters:
        xci, zci : array_like
            Coordinates of influenced points.
        xcj, zcj : array_like
            Coordinates of influencing panel control points.
        tauxj, tauzj : array_like
            Unit tangent vectors for the influencing panels.
        nxj, nzj : array_like
            Unit normal vectors for the influencing panels.
        sj : array_like
            Lengths of the influencing panels.

    Returns:
        Ausigma, Avsigma, Augamma, Avgamma : ndarray
            Influence coefficient matrices.
    """
    # Transform the influencing coordinates to arrays (useful if only one panel
    # is influencing)
    xcj = np.array([xcj])
    zcj = np.array([zcj])
    tauxj = np.array([tauxj])
    tauzj = np.array([tauzj])
    nxj = np.array([nxj])
    nzj = np.array([nzj])
    sj = np.array([sj])

    ni = len(xci)
    nj = len(xcj)

    # Initialize influence coefficient matrices
    Ausigma = np.zeros((ni, nj))
    Avsigma = np.zeros((ni, nj))

    # Cycle over influenced points
    for i in range(ni):
        # Cycle over influencing panels
        for j in range(nj):
            # Calculate distance between influenced point and control point of influencing panel
            rx = xci[i] - xcj[j]  # Horizontal distance
            rz = zci[i] - zcj[j]  # Vertical distance
            rdist = np.sqrt(rx**2 + rz**2)  # Total distance

            if rdist < 1e-8:
                # Self-influence: the influenced point lies on the control point of the influencing panel
                utau = 0  # Tangential velocity influence
                vn = 0.5  # Normal velocity influence
            else:
                # Tangential distance between influenced point and control point of influencing panel
                xtau = rx * tauxj[j] + rz * tauzj[j]
                # Normal distance between influenced point and control point of influencing panel
                zn = rx * nxj[j] + rz * nzj[j]
                # Tangential source velocity influence, equation 4.114
                utau = (
                    -1
                    / (4 * np.pi)
                    * np.log(
                        ((xtau - sj[j] / 2) ** 2 + zn**2)
                        / ((xtau + sj[j] / 2) ** 2 + zn**2)
                    )
                )
                # Normal source velocity influence, equation 4.115
                vn = (
                    1
                    / (2 * np.pi)
                    * (
                        np.arctan2(zn, (xtau - sj[j] / 2))
                        - np.arctan2(zn, (xtau + sj[j] / 2))
                    )
                )

            # Project tangential and normal velocity influences onto x and y axes
            # Horizontal velocity influence coefficient matrix
            Ausigma[i, j] = utau * tauxj[j] + vn * nxj[j]
            # Vertical velocity influence coefficient matrix
            Avsigma[i, j] = utau * tauzj[j] + vn * nzj[j]

    # Calculate vortex influence coefficient matrices
    Augamma = Avsigma.copy()
    Avgamma = -Ausigma

    return Augamma, Avgamma


def vpminf(xci, zci, xcj, zcj, tauxj, tauzj, nxj, nzj, sj):
    """
    Influence coefficient matrices for the vortex panel method with linearly
    varying strength. Calculates the horizontal and vertical velocity influence
    of source and vortex panel j on point i.

    Parameters:
    xci, zci : array-like
        Coordinates of influenced points.
    xcj, zcj : array-like
        Control points of influencing panels.
    tauxj, tauzj : array-like
        Unit tangent vectors of influencing panels.
    nxj, nzj : array-like
        Unit normal vectors of influencing panels.
    sj : array-like
        Lengths of influencing panels.

    Returns:
    Au : ndarray
        Horizontal velocity influence coefficient matrix.
    Av : ndarray
        Vertical velocity influence coefficient matrix.

    Written by Grigorios Dimitriadis, May 2023.
    This code accompanies the book Unsteady Aerodynamics - Potential and
    Vortex Methods, G. Dimitriadis, Wiley, 2023. If any part of the code is
    reused, the author and book must be acknowledged.
    """
    # Transform the influenced coordinates to arrays (useful if only one panel
    # is influenced)
    if isinstance(xci, (np.floating, float)):
        xci = np.array([xci])
        zci = np.array([zci])

    ni = len(xci)
    nj = len(xcj)

    # Initialize influence coefficient matrices
    Au1 = np.zeros((ni, nj))
    Au2 = np.zeros((ni, nj))
    Av1 = np.zeros((ni, nj))
    Av2 = np.zeros((ni, nj))

    # Cycle over influenced points
    for i in range(ni):
        # Cycle over influencing panels
        for j in range(nj):
            # Calculate distance between influenced point and control point of influencing panel
            rx = xci[i] - xcj[j]  # Horizontal distance
            rz = zci[i] - zcj[j]  # Vertical distance
            rdist = np.sqrt(rx**2 + rz**2)  # Total distance

            # Tangential distance between influenced point and control point of influencing panel
            xtau = rx * tauxj[j] + rz * tauzj[j]

            # Normal distance between influenced point and control point of influencing panel
            zn = rx * nxj[j] + rz * nzj[j]

            # Calculate log and atan terms
            if rdist < 1e-8:
                # Self-influence: the influenced point lies on the control point of the influencing panel
                logterm = 0
                atanterm = np.pi
            else:
                # Use this form of the atan terms to be able to apply the atan2 function.
                atanterm = np.arctan2(zn, (xtau - sj[j] / 2)) - np.arctan2(
                    zn, (xtau + sj[j] / 2)
                )
                # Logarithmic term
                logterm = np.log(
                    ((xtau - sj[j] / 2) ** 2 + zn**2)
                    / ((xtau + sj[j] / 2) ** 2 + zn**2)
                )

            # Calculate velocities at point i in a direction tangent to panel j (equation 4.189)
            utau1 = (
                1
                / (2 * np.pi * -sj[j])
                * (atanterm * (xtau - sj[j] / 2) + zn * logterm / 2)
            )

            # Calculate velocities at point i in a direction tangent to panel j (equation 4.190)
            utau2 = (
                -1
                / (2 * np.pi * -sj[j])
                * (atanterm * (xtau + sj[j] / 2) + zn * logterm / 2)
            )

            # Calculate velocities at point i in a direction normal to panel j (equation 4.191)
            vn1 = (
                1
                / (2 * np.pi * -sj[j])
                * (sj[j] - zn * atanterm + logterm * (xtau - sj[j] / 2) / 2)
            )

            # Calculate velocities at point i in a direction normal to panel j (equation 4.192)
            vn2 = (
                -1
                / (2 * np.pi * -sj[j])
                * (sj[j] - zn * atanterm + logterm * (xtau + sj[j] / 2) / 2)
            )

            # Calculate velocities at point i in x direction (equation 4.195)
            Au1[i, j] = utau1 * tauxj[j] + vn1 * nxj[j]
            Au2[i, j] = utau2 * tauxj[j] + vn2 * nxj[j]

            # Calculate velocities at point i in y direction (equation 4.195)
            Av1[i, j] = utau1 * tauzj[j] + vn1 * nzj[j]
            Av2[i, j] = utau2 * tauzj[j] + vn2 * nzj[j]

    # Calculate horizontal velocity influence coefficients (equation 4.193)
    Au = np.hstack([Au1, np.zeros((ni, 1))]) + np.hstack([np.zeros((ni, 1)), Au2])

    # Calculate vertical velocity influence coefficients (equation 4.194)
    Av = np.hstack([Av1, np.zeros((ni, 1))]) + np.hstack([np.zeros((ni, 1)), Av2])

    return Au, Av
