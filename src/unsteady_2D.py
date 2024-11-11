import numpy as np
import matplotlib.pyplot as plt
from .shared.utils import Read_Two_Column_File, flip_panels, rotate_points_airfoil


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
    airfoil_name,
    airfoil_type,
    flap_position,
    beta_input=0,
    desired_chord=1,
    theta_input=0,
    x_pitch=0,
    z_pitch=0,
    skiprows=0,
    n=400,
):
    """
    Function to generate the geometry of an airfoil using the NACA four-digit
    series. The airfoil is discretized into a number of panels, and the
    coordinates of the panel nodes, control points, and panel angles are
    calculated.

    Inputs:
    - airfoil_name: name of the airfoil file
    - flap_position: position of the flap hinge
    - theta_input: angle of rotation of the airfoil [rad]
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
    # if n not even, throw an error
    if n % 2 != 0:
        raise ValueError("Number of panels (n) must be even")

    if airfoil_type == "naca":
        # Define the NACA reference number and number of panels
        mpt = airfoil_name  # NACA reference number

        # Set non-dimensional x-coordinates of the airfoil
        x_int = 0.5 * (
            1 - np.cos(np.linspace(-np.pi, np.pi, n + 1))
        )  # Nonlinear distribution

        # Calculate non-dimensional y-coordinates of the airfoil
        z_int = nacafourdigit(x_int, int(n / 2), mpt, 1)
    else:
        x_int, z_int = Read_Two_Column_File(airfoil_name, skiprows)

    # Flip the panels if needed
    (x_int, z_int) = flip_panels(x_int, z_int, "CW")

    # Smoothen the airfoil
    # TO DO

    # Scale the airfoil according to the desired chord
    original_chord = np.amax(x_int) - np.amin(x_int)
    x_int *= desired_chord / original_chord
    z_int *= desired_chord / original_chord

    # Shift the reference system to the pitching axis
    x_int -= x_pitch
    z_int -= z_pitch
    flap_position -= np.array([x_pitch, z_pitch])

    # Identify panel nodes that need to be rotated
    x_hinge = flap_position[0]
    z_hinge = flap_position[1]
    indexes_flap = np.where(x_int >= x_hinge)

    # Rotate points according to the flap angle
    x_int, z_int = rotate_points_airfoil(
        x_int, z_int, x_hinge, z_hinge, beta_input, indexes_flap
    )

    # Rotate the airfoil according to the current angle of attack
    R_theta = np.array(
        [
            [np.cos(theta_input), np.sin(theta_input)],
            [-np.sin(theta_input), np.cos(theta_input)],
        ]
    )
    xzrot = np.dot(R_theta, np.array([x_int, z_int]))

    x_int = xzrot[0, :]
    z_int = xzrot[1, :]

    flap_position = np.dot(R_theta, flap_position)

    numPan = len(x_int) - 1  # Number of panels

    # Calculate control points
    xc = x_int[:-1] + (x_int[1:] - x_int[:-1]) / 2
    zc = z_int[:-1] + (z_int[1:] - z_int[:-1]) / 2

    # Calculate panel lengths, tangent and normal vectors
    s = np.sqrt((x_int[1:] - x_int[:-1]) ** 2 + (z_int[1:] - z_int[:-1]) ** 2)
    tx = (x_int[1:] - x_int[:-1]) / s
    tz = (z_int[1:] - z_int[:-1]) / s
    nx = -tz
    nz = tx

    indexes_flap_c = np.where(xc > x_hinge)

    return (
        x_int,
        z_int,
        xc,
        zc,
        numPan,
        s,
        nx,
        nz,
        tx,
        tz,
        indexes_flap_c,
        flap_position,
    )


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


def vpminf(xci, zci, xcj, zcj, tauxj, tauzj, nxj, nzj, sj, epsilon=1e-10):
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
    epsilon : float, optional
        Small number to avoid division by zero.

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
                    ((xtau - sj[j] / 2) ** 2 + zn**2 + epsilon)
                    / ((xtau + sj[j] / 2) ** 2 + zn**2 + epsilon)
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


def compute_aerodynamics_static(aero_input_list):
    airfoil_name = aero_input_list["file_airfoil"]
    airfoil_type = aero_input_list["type_airfoil"]
    x_beta = aero_input_list["x_flap"]
    z_beta = aero_input_list["z_flap"]
    beta_val = aero_input_list["flap_angle"]
    c = aero_input_list["chord"]
    airfoil_skiprows = aero_input_list["number_skiprows"]
    alpha_val = aero_input_list["alpha_input"]
    U_inf = aero_input_list["freestream_velocity"]
    # Update geometry
    (
        x_airfoil,
        z_airfoil,
        xc,
        zc,
        N,
        s,
        nx,
        nz,
        tx,
        tz,
        indexes_flap,
        hinge_position,
    ) = geometry_airfoil(
        airfoil_name,
        airfoil_type,
        np.array([x_beta, z_beta]),
        beta_input=beta_val,
        desired_chord=c,
        skiprows=airfoil_skiprows,
    )
    # Determine coefficient matrix
    [Au, Av] = vpminf(xc, zc, xc, zc, tx, tz, nx, nz, s)
    coeff_matrix = Au * nx[:, np.newaxis] + Av * nz[:, np.newaxis]
    Atau = Au * tx[:, np.newaxis] + Av * tz[:, np.newaxis]
    # Assemble the RHS vector (without the Kutta condition)
    RHS = -(U_inf * np.cos(alpha_val) * nx + U_inf * np.sin(alpha_val) * nz)
    # Add Kutta condition to the coefficient matrix
    A_Kutta = np.zeros(N + 1)
    A_Kutta = Atau[0, :] + Atau[-1, :]
    coeff_matrix = np.vstack((coeff_matrix, A_Kutta))
    # Add Kutta condition to the RHS vector
    btau = U_inf * np.cos(alpha_val) * tx + U_inf * np.sin(alpha_val) * tz
    RHS = np.append(RHS, -btau[0] - btau[-1])

    # Solve the system of equations
    gamma = np.linalg.solve(coeff_matrix, RHS)

    # Compute the pressure coefficient and the normal force
    utang = np.dot(Atau, gamma) + btau
    C_p = 1 - utang**2 / U_inf**2
    cni = -C_p * s * nz / c
    c_n = np.sum(cni)

    return (C_p, c_n)


def compute_aerodynamics_unsteady(aero_input_list):
    airfoil_skiprows = aero_input_list["number_skiprows"]
    airfoil_name = aero_input_list["file_airfoil"]
    airfoil_type = aero_input_list["type_airfoil"]
    flapping_bool = aero_input_list["flap_or_not"]
    pitching_bool = aero_input_list["pitch_or_not"]
    flap_characteristics = aero_input_list["flapping_parameters"]
    pitch_characteristics = aero_input_list["pitching_parameters"]
    c = aero_input_list["chord"]
    k_beta = aero_input_list["reduced_frequency_flap"]
    k_theta = aero_input_list["reduced_frequency_pitch"]
    U_inf = aero_input_list["U_inf"]
    W_inf = aero_input_list["W_inf"]
    dt_per_cycle = aero_input_list["timesteps_per_period"]
    total_N_cycles = aero_input_list["total_cycles"]
    N_cycles_startup = aero_input_list["startup_cycles"]
    N = aero_input_list["number_panels"]
    rho = aero_input_list["density"]
    WAKE_MODEL = aero_input_list["wake_type"]
    STAGNATION_PHI_CHOICE = aero_input_list["stagnation_potential_location"]
    AERO_MODEL = aero_input_list["aerodynamic_model"]
    max_N_iter_NVPM = aero_input_list["iterations_Newton_TE_panel"]
    rel_change_NVPM = aero_input_list["relative_change_crit_Newton_TE_panel"]
    dx = aero_input_list["dx_Newton_TE_panel"]

    beta_0 = flap_characteristics[0]
    beta_mean = flap_characteristics[1]
    phi_beta = flap_characteristics[2]
    x_beta = flap_characteristics[3]
    z_beta = flap_characteristics[4]

    theta_0 = pitch_characteristics[0]
    theta_mean = pitch_characteristics[1]
    phi_theta = pitch_characteristics[2]
    x_theta = pitch_characteristics[3]
    z_theta = pitch_characteristics[4]

    # For defining dynamic movements of the airfoil when it is both pitching and
    # flapping, we will assume the reduced frequency of the pitching motion to
    # take precedence
    if pitching_bool:
        omega_theta = 2 * k_theta * U_inf / c  # rad/s
    else:
        k_theta = 0
        omega_theta = 0

    if flapping_bool:
        omega_beta = 2 * k_beta * U_inf / c  # rad/s
    else:
        k_beta = 0
        omega_beta = 0

    omega = omega_theta if pitching_bool else omega_beta
    k = k_theta if pitching_bool else k_beta

    # Set up reduced frequency, timestepping
    t_array = np.linspace(
        0,
        total_N_cycles * 2 * np.pi / omega,
        int(dt_per_cycle * total_N_cycles + 1),
    )
    dt = t_array[1] - t_array[0]
    N_timesteps = t_array.size

    # Initialize the dynamic velocity arrays
    U_array = np.ones(t_array.size) * U_inf
    W_array = np.ones(t_array.size) * W_inf

    # Initialize circulation and positions of the wake vortices
    gamma_w = np.zeros(N_timesteps)
    xw = np.zeros(N_timesteps)
    zw = np.zeros(N_timesteps)
    xw_array = np.zeros((N_timesteps, N_timesteps))
    zw_array = np.zeros((N_timesteps, N_timesteps))
    # Initialize airfoil position
    x_airfoil_array = np.zeros((N + 1, N_timesteps))
    z_airfoil_array = np.zeros((N + 1, N_timesteps))

    # Initialize values of the pitch angle and the pitch rate (theta_store
    # and theta_dot_store)
    theta_store = theta_mean + theta_0 * np.sin(
        omega_theta * (t_array - N_cycles_startup * dt) + phi_theta
    )
    beta_store = beta_mean + beta_0 * np.sin(
        omega_beta * (t_array - N_cycles_startup * dt) + phi_beta
    )

    theta_dot_store = (
        theta_0
        * omega_theta
        * np.cos(omega_theta * (t_array - N_cycles_startup * dt) + phi_theta)
    )
    beta_dot_store = (
        beta_0
        * omega_beta
        * np.cos(omega_beta * (t_array - N_cycles_startup * dt) + phi_beta)
    )
    beta_store[0 : int(N_cycles_startup * dt_per_cycle) + 1] = beta_mean + phi_beta
    theta_dot_store[0 : int(N_cycles_startup * dt_per_cycle) + 1] = 0
    beta_dot_store[0 : int(N_cycles_startup * dt_per_cycle) + 1] = 0

    # Total bound circulation around the airfoil
    gamma_old = 0

    # Initialize wake-panel related quantitites
    # Angle
    theta_wp = 0
    # Length
    l_wp = dt * U_array[0]
    # Initial vortex strength
    gamma_wp = 0
    # Initialize velocities on the wake panel's control point
    uw_wp = 0
    ww_wp = 0
    # Arrays of wake panel lengths and angles
    l_wp_array = np.zeros(N_timesteps)
    theta_wp_array = np.zeros(N_timesteps)

    # Initialize matrices to store motion-induced velocities on the control
    # points
    u_m = np.zeros((N, N_timesteps))
    w_m = np.zeros((N, N_timesteps))

    # Initialize matrices to store velocities induced by the wake vortices
    # on the control points
    u_w = np.zeros((N, N_timesteps))
    w_w = np.zeros((N, N_timesteps))

    # Previous values of the potential at the previous timestep
    phi_c_old = np.zeros(N)

    lift_array = np.zeros(N_timesteps)
    drag_array = np.zeros(N_timesteps)
    C_p_array = np.zeros((N, N_timesteps))
    C_n_array = np.zeros(N_timesteps)
    for i, t in enumerate(t_array):
        # First, compute the known geometry and motion-related
        # characteristics for the current timestep
        U_inf_t = U_array[i]
        W_inf_t = W_array[i]

        Q_inf_t = np.sqrt(U_inf_t**2 + W_inf_t**2)

        theta = theta_store[i]
        theta_dot = theta_dot_store[i]

        beta = beta_store[i]
        beta_dot = beta_dot_store[i]

        (
            x_airfoil,
            z_airfoil,
            xc,
            zc,
            N,
            s,
            nx,
            nz,
            tx,
            tz,
            indexes_flap,
            hinge_position,
        ) = geometry_airfoil(
            airfoil_name,
            airfoil_type,
            np.array([x_beta, z_beta]),
            beta_input=beta,
            desired_chord=c,
            theta_input=theta,
            x_pitch=x_theta,
            z_pitch=z_theta,
            skiprows=airfoil_skiprows,
        )
        x_hinge = hinge_position[0]
        z_hinge = hinge_position[1]

        [Au, Av] = vpminf(xc, zc, xc, zc, tx, tz, nx, nz, s)
        coeff_matrix = Au * nx[:, np.newaxis] + Av * nz[:, np.newaxis]
        Atau = Au * tx[:, np.newaxis] + Av * tz[:, np.newaxis]

        # Add the effect of the airfoil motion to the right-hand side of
        # the impermeability condition
        # Add effect of pitchingx
        u_m[:, i] = U_inf_t - theta_dot * zc
        w_m[:, i] = theta_dot * xc
        if flapping_bool:
            # Add effect of flapping
            u_m[indexes_flap, i] -= beta_dot * (zc[indexes_flap] - z_hinge)
            w_m[indexes_flap, i] += beta_dot * (xc[indexes_flap] - x_hinge)

        b_vec = -u_m[:, i] * nx - w_m[:, i] * nz
        b_tau = u_m[:, i] * tx + w_m[:, i] * tz

        # Add the effect of the shed vorticity to the right-hand side of the
        # impermeability condition
        if i > 0:
            # Convect the wake vortices away from the airfoil (use velocities
            # computed at previous time step)
            if i > 1:
                gamma_w[1 : i + 1] = gamma_w[0:i]
                U_inf_t_m1 = U_array[i - 1]
                W_inf_t_m1 = W_array[i - 1]
                xw[1 : i + 1] = xw[0:i] + (U_inf_t_m1 + u_w_prop) * dt
                zw[1 : i + 1] = zw[0:i] + (W_inf_t_m1 + w_w_prop) * dt

            # Include the effect of the wake panel (for this, we need to
            # compute the midpoints of the wake panel and the circulation;
            # once this is done, we compute the influence over each bound
            # panel's node)
            gamma_w[0] = gamma_wp * l_wp
            if WAKE_MODEL == "straight":
                xw[0] = (c - x_theta) + 0.5 * l_wp
                zw[0] = 0
            elif WAKE_MODEL == "prescribed":
                xw[0] = xc_wp + U_inf_t * dt
                zw[0] = zc_wp + W_inf_t * dt
            elif WAKE_MODEL == "free-wake":
                xw[0] = xc_wp + u_wp[0] * dt
                zw[0] = zc_wp + w_wp[0] * dt

            B_u, B_v = lvminf(xc, zc, xw[0:i], zw[0:i])
            u_w[:, i] = np.dot(B_u, gamma_w[0:i])
            w_w[:, i] = np.dot(B_v, gamma_w[0:i])

        # Determine the normal and tangential components of wake-induced
        # velocities
        v_nw = u_w[:, i] * nx + w_w[:, i] * nz
        u_tauw = u_w[:, i] * tx + w_w[:, i] * tz

        # Based on the velocities induced by the motion and the wake, iterate
        # for the wake panel length, angle, and circulation
        wake_iter = 1
        n_iter_wp = 0

        # Build dictionary for the newtonVPM function
        newtonVPM_dict = {
            "lwp": l_wp,
            "thetawp": theta_wp,
            "x": x_airfoil,
            "z": z_airfoil,
            "xc": xc,
            "zc": zc,
            "xw": xw,
            "zw": zw,
            "Gammaw": gamma_w,
            "nx": nx,
            "nz": nz,
            "tx": tx,
            "tz": tz,
            "s": s,
            "U": U_inf_t,
            "dt": dt,
            "uw_wp": uw_wp,
            "ww_wp": ww_wp,
            "An": coeff_matrix,
            "Atau": Atau,
            "bvec": b_vec,
            "btau": b_tau,
            "utauw": u_tauw,
            "vnw": v_nw,
            "Gamma_old": gamma_old,
            "it": i,
        }
        while wake_iter == 1:
            n_iter_wp += 1

            # Calculate the objective function associated with the
            # Newton-Raphson search method
            output_NVPM = newtonVPM(newtonVPM_dict)
            Fsv = output_NVPM["Fsv"]
            # Calculatethe Jacobian
            Jac = np.zeros((2, 2))

            # Derivatives with respect to lwp
            l_wp += dx
            newtonVPM_dict["lwp"] = l_wp
            output_NVPM_plus_lwp = newtonVPM(newtonVPM_dict)
            Fsv_plus_lwp = output_NVPM_plus_lwp["Fsv"]
            Jac[:, 0] = np.squeeze((Fsv_plus_lwp - Fsv) / dx)
            l_wp -= dx
            newtonVPM_dict["lwp"] = l_wp

            # Derivatives with respect to thetawp
            theta_wp += dx
            newtonVPM_dict["thetawp"] = theta_wp
            output_NVPM_plus_thetawp = newtonVPM(newtonVPM_dict)
            Fsv_plus_thetawp = output_NVPM_plus_thetawp["Fsv"]
            Jac[:, 1] = np.squeeze((Fsv_plus_thetawp - Fsv) / dx)
            theta_wp -= dx
            newtonVPM_dict["thetawp"] = theta_wp

            # Solve the Newton-Raphson system
            delta_x = -np.linalg.solve(Jac, Fsv)

            # Calculate convergence criterion
            crit = np.sqrt(delta_x[0] ** 2 + delta_x[1] ** 2)

            # Assign updated values to the wake panel and length
            l_wp += delta_x[0][0]
            newtonVPM_dict["lwp"] = l_wp

            theta_wp += delta_x[1][0]
            newtonVPM_dict["thetawp"] = theta_wp

            # Test for convergence
            if crit < rel_change_NVPM:
                wake_iter = 0
            if n_iter_wp > max_N_iter_NVPM:
                print("exceeded maximum number of iterations for Newton-Raphson")
                l_wp = l_wp_array[i - 1]
                theta_wp = theta_wp_array[i - 1]
                wake_iter = 0

        # Calculate converged values associated with the wake panel
        last_iteration_output_NVPM = newtonVPM(newtonVPM_dict)

        x_wp = last_iteration_output_NVPM["xwp"]
        z_wp = last_iteration_output_NVPM["zwp"]
        xc_wp = last_iteration_output_NVPM["xcwp"]
        zc_wp = last_iteration_output_NVPM["zcwp"]
        nx_wp = last_iteration_output_NVPM["nxwp"]
        nz_wp = last_iteration_output_NVPM["nzwp"]
        tx_wp = last_iteration_output_NVPM["tauxwp"]
        tz_wp = last_iteration_output_NVPM["tauzwp"]
        gamma_wp = last_iteration_output_NVPM["gammawp"]
        gamma_bound = last_iteration_output_NVPM["gamma"]
        u_wp = last_iteration_output_NVPM["uwp"]
        w_wp = last_iteration_output_NVPM["wwp"]
        u_tan_bound = last_iteration_output_NVPM["utang"]

        # Store converged angle and length of wake panel
        theta_wp_array[i] = theta_wp
        l_wp_array[i] = l_wp

        # Calculate value of total bound circulation
        s_bar = (np.concatenate([[0], s]) + np.concatenate([s, [0]])) / 2
        gamma_old = np.sum(gamma_bound * s_bar)

        # Calculate effect of wake panel on the shed vorticity
        u_w_prop = np.zeros(i + 1)
        w_w_prop = np.zeros(i + 1)
        if i > 0:
            if WAKE_MODEL == "free-wake":
                Au_w, Aw_w = vpminf(
                    xw[0 : i + 1], zw[0 : i + 1], xc, zc, tx, tz, nx, nz, s
                )
                u_w_prop += np.dot(Au_w, gamma_bound)
                w_w_prop += np.dot(Aw_w, gamma_bound)

                Au_wp, Aw_wp = svpminf(
                    xw[0 : i + 1],
                    zw[0 : i + 1],
                    xc_wp,
                    zc_wp,
                    tx_wp,
                    tz_wp,
                    nx_wp,
                    nz_wp,
                    l_wp,
                )
                Au_wp = Au_wp.flatten()
                Aw_wp = Aw_wp.flatten()

                u_w_prop += Au_wp * gamma_wp
                w_w_prop += Aw_wp * gamma_wp

                B_uw, B_wv = lvminf(
                    xw[0 : i + 1],
                    zw[0 : i + 1],
                    xw[0 : i + 1],
                    zw[0 : i + 1],
                )

                u_w_prop += np.dot(B_uw, gamma_w[0 : i + 1])
                w_w_prop += np.dot(B_wv, gamma_w[0 : i + 1])

            # Calculation of perturbation potential; first, calculate potential
            # at leading edge by settings up grid upstream of the airfoil in
            # the direction of the freestream
            N_chords = 10
            index_panel_stagnation, phi_LE = compute_stagnation_phi(
                x_airfoil,
                z_airfoil,
                s,
                STAGNATION_PHI_CHOICE,
                AERO_MODEL,
                N_chords,
                xc,
                zc,
                tx,
                tz,
                nx,
                nz,
                c,
                gamma_bound,
                i,
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
            )

            # Calculate the potential
            phi_airfoil = np.zeros(N + 1)

            # Lower surface
            for j in range(index_panel_stagnation):
                phi_airfoil[j + 1] = phi_LE - np.sum(
                    u_tan_bound[index_panel_stagnation:j:-1]
                    * s[index_panel_stagnation:j:-1]
                )
            phi_airfoil[0] = phi_LE - np.sum(
                u_tan_bound[index_panel_stagnation::-1] * s[index_panel_stagnation::-1]
            )
            # Upper surface
            for j in range(index_panel_stagnation + 2, N + 1):
                phi_airfoil[j] = phi_LE + np.sum(
                    u_tan_bound[index_panel_stagnation + 1 : j]
                    * s[index_panel_stagnation + 1 : j]
                )
            # Stagnation point
            phi_airfoil[index_panel_stagnation + 1] = phi_LE

            # Potential at the control points
            phi_c = 0.5 * (phi_airfoil[0:-1] + phi_airfoil[1:])
            # Derivative of the potential
            if i == 0:
                d_phic_dt = np.zeros(N)
            else:
                d_phic_dt = (phi_c - phi_c_old) / dt
            phi_c_old = phi_c

            # Compute the pressure coefficient and aerodynamic loads
            C_p = (
                1
                - ((b_tau + u_tan_bound) ** 2) / (Q_inf_t**2)
                - 2 / Q_inf_t**2 * d_phic_dt
            )
            cni = -C_p * s * nz / c
            C_n_array[i] = np.sum(cni)
            C_p_array[:, i] = C_p
            xw_array[:, i] = xw
            zw_array[:, i] = zw
            x_airfoil_array[:, i] = x_airfoil
            z_airfoil_array[:, i] = z_airfoil
            lift_array[i] = -0.5 * rho * U_inf_t**2 * np.sum(C_p * s * nz)
            drag_array[i] = -0.5 * rho * U_inf_t**2 * np.sum(C_p * s * nx)

    unsteady_aero_output_dict = {
        "time_instances": t_array,
        "theta_values": theta_store,
        "beta_values": beta_store,
        "lift": lift_array,
        "drag": drag_array,
        "C_p": C_p_array,
        "C_n": C_n_array,
        "x_airfoil": x_airfoil_array,
        "z_airfoil": z_airfoil_array,
        "x_wake": xw_array,
        "z_wake": zw_array,
        "N_timesteps": N_timesteps,
        "length_TE_panel": l_wp_array,
        "angle_TE_panel": theta_wp_array,
    }

    return unsteady_aero_output_dict
