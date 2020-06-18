import numpy as np
import math
from mpl_toolkits.mplot3d import art3d
from matplotlib import colors as mpl_colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def make_patch_3d(pathpatch, rot_axis, angle, z=0):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    path = pathpatch.get_path()  # Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path)  # Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D  # Change the class
    pathpatch._code3d = path.codes  # Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor  # Get the face color

    verts = path.vertices  # Get the vertices in 2D

    M = _rotation_matrix(rot_axis, angle)  # Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])


def pathpatch_2d_to_3d(pathpatch, z=0, normal='z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str:  # Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0, 0, 0), index)

    normal /= np.linalg.norm(normal)  # Make sure the vector is normalised

    path = pathpatch.get_path()  # Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path)  # Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D  # Change the class
    pathpatch._code3d = path.codes  # Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor  # Get the face color

    verts = path.vertices  # Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1))  # Obtain the rotation vector
    M = _rotation_matrix_sin(d)  # Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])


def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta


def get_polygons_for_cylinder(pos_start, direction, length, radius, n_points, facecolor='b', edgecolor='k', alpha=1.,
                              lw=0.):
    """
    Returns polygons from a cylinder

    Parameters
    ----------
    pos_start: np.array
        3D position of the bottom of the cylinder
    direction: np.array
        Cylinder direction
    length: float
        Length of the cylinder
    radius: float
        Radius of the cylinder
    n_points: int
        Number of points for polygons
    facecolor: Matplolib color
        Color of the faces
    edgecolor: Matplolib color
        Color of the edges
    alpha: float
        Alpha value
    lw: float
        Linewidth of the edges

    Returns
    -------
    collection: Matplotlib rd cllection
        Collection of polygons making up the cylinder
    """
    n_points = int(n_points)
    x, y, z = _cylinder(pos_start,
                        direction,
                        length,
                        radius,
                        n_points)

    alpha_tup = alpha,
    edge_col = mpl_colors.to_rgb(edgecolor) + alpha_tup
    face_col = mpl_colors.to_rgb(facecolor) + alpha_tup

    theta_ring = np.linspace(0., np.pi * 2., n_points)
    verts_hull = []
    for idx_theta, crt_theta in enumerate(theta_ring):
        if idx_theta <= theta_ring.size - 2:
            x_verts = [x[idx_theta],
                       x[idx_theta + 1],
                       x[idx_theta + 1 + theta_ring.size],
                       x[idx_theta + theta_ring.size]]
            y_verts = [y[idx_theta],
                       y[idx_theta + 1],
                       y[idx_theta + 1 + theta_ring.size],
                       y[idx_theta + theta_ring.size]]
            z_verts = [z[idx_theta],
                       z[idx_theta + 1],
                       z[idx_theta + 1 + theta_ring.size],
                       z[idx_theta + theta_ring.size]]
            verts_hull.append(zip(x_verts, y_verts, z_verts))

    poly3d_hull = []
    for crt_vert in verts_hull:
        cyl = Poly3DCollection([list(crt_vert)], linewidths=lw)
        cyl.set_facecolor(face_col)
        cyl.set_edgecolor(edge_col)
        cyl.set_alpha(alpha)

        poly3d_hull.append(cyl)

    # draw lower lid
    x_verts = x[0:theta_ring.size - 1]
    y_verts = y[0:theta_ring.size - 1]
    z_verts = z[0:theta_ring.size - 1]
    verts_lowerlid = [list(zip(x_verts, y_verts, z_verts))]
    poly3ed_lowerlid = Poly3DCollection(verts_lowerlid, linewidths=lw, zorder=1)
    poly3ed_lowerlid.set_facecolor(face_col)
    poly3ed_lowerlid.set_edgecolor(edge_col)
    poly3ed_lowerlid.set_alpha(alpha)

    # draw upper lid
    x_verts = x[theta_ring.size:theta_ring.size * 2 - 1]
    y_verts = y[theta_ring.size:theta_ring.size * 2 - 1]
    z_verts = z[theta_ring.size:theta_ring.size * 2 - 1]
    verts_upperlid = [list(zip(x_verts, y_verts, z_verts))]
    poly3ed_upperlid = Poly3DCollection(verts_upperlid, linewidths=lw, zorder=1)
    poly3ed_upperlid.set_facecolor(face_col)
    poly3ed_upperlid.set_edgecolor(edge_col)
    poly3ed_upperlid.set_alpha(alpha)

    return_col = poly3d_hull
    return_col.append(poly3ed_lowerlid)
    return_col.append(poly3ed_upperlid)

    return return_col


##### from plotting convention ######
def mark_subplots(axes, letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ', xpos=-0.05, ypos=1.05, fs=50):
    """
    Marks axes with letters

    Parameters
    ----------
    axes: list
        List of Matplotlib axes to mark
    letters: list
        List of sequential characters (default alphabet)
    xpos: float
        x-position with respect to each axis (default -0.05)
    ypos: float
        y-position with respect to each axis (default 1.05)
    fs: int
        Fontsize of the letters

    """
    if not type(axes) is list:
        axes = [axes]

    for idx, ax in enumerate(axes):
        # Axes3d
        try:
            ax.text2D(xpos, ypos, letters[idx].capitalize(),
                      horizontalalignment='center',
                      verticalalignment='center',
                      fontweight='demibold',
                      fontsize=fs,
                      transform=ax.transAxes)
        except AttributeError:
            ax.text(xpos, ypos, letters[idx].capitalize(),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontweight='demibold',
                    fontsize=fs,
                    transform=ax.transAxes)


def simplify_axes(axes):
    """
    Removes top and right spines from axes

    Parameters
    ----------
    axes: list
        List of Matplotlib axes to simplify
    """
    if not type(axes) is list:
        axes = [axes]

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()


def color_axes(axes, color):
    """
    Colors axes spines

    Parameters
    ----------
    axes: list
        List of Matplotlib axes to simplify
    color: Matplotlib color
        The color to be used
    """
    if not type(axes) is list:
        axes = [axes]
    for ax in axes:
        ax.tick_params(axis='x', colors=color)
        ax.tick_params(axis='y', colors=color)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)


####### HELPER #########
def _rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / np.linalg.norm(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def _rotation_matrix_sin(d):
    """
    Calculates a rotation matrix given a vector d. The direction of d
    corresponds to the rotation axis. The length of d corresponds to
    the sin of the angle of rotation.

    Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    """
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        return np.identity(3)

    d /= sin_angle
    sin_angle = np.linalg.norm(d)

    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[0, d[2], -d[1]],
                     [-d[2], 0, d[0]],
                     [d[1], -d[0], 0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle ** 2) * (eye - ddt) + sin_angle * skew
    return M


def _cylinder(pos_start, direction, length, radius, n_points):
    # Build cylinder pointing to positive x-direction with right radius and length
    alpha = np.array([0., length])
    theta_ring = np.linspace(0., np.pi * 2., int(n_points))
    r = radius

    x = np.zeros((theta_ring.size * alpha.size))
    y = np.zeros((theta_ring.size * alpha.size))
    z = np.zeros((theta_ring.size * alpha.size))

    for idx_alpha, crt_alpha in enumerate(alpha):
        x[idx_alpha * theta_ring.size:(idx_alpha + 1) * theta_ring.size] = r * np.cos(theta_ring)
        y[idx_alpha * theta_ring.size:(idx_alpha + 1) * theta_ring.size] = r * np.sin(theta_ring)
        z[idx_alpha * theta_ring.size:(idx_alpha + 1) * theta_ring.size] = crt_alpha * np.ones(theta_ring.size)

    # rotate cylinder to match direction
    #
    # - rho: length of vector projection on x-y plane
    # - phy: angle of rotation wrt y-axis
    # - theta: angle of rotation wrt z-axis

    y_axis = np.array([0., 1., 0.])
    z_axis = np.array([0., 0., 1.])

    if direction[0] == 0:
        theta = -np.sign(direction[1]) * np.pi / 2.
    else:
        if direction[0] > 0:
            theta = -np.arctan(direction[1] / direction[0])
        else:
            theta = np.pi - np.arctan(direction[1] / direction[0])

    rho = np.sqrt((direction[0] ** 2 + direction[1] ** 2))
    if rho == 0:
        if direction[2] > 0:
            phi = 0.
        else:
            phi = np.pi
    else:
        phi = -(np.pi / 2. - np.arctan(direction[2] / rho))

    rot1_m = _rotation_matrix(y_axis, phi)
    rot2_m = _rotation_matrix(z_axis, theta)

    for idx, (crt_x, crt_y, crt_z) in enumerate(zip(x, y, z)):
        crt_v = np.array([crt_x, crt_y, crt_z])
        crt_v = np.dot(crt_v, rot1_m)
        crt_v = np.dot(crt_v, rot2_m)
        x[idx] = crt_v[0]
        y[idx] = crt_v[1]
        z[idx] = crt_v[2]

    # move cylinder to start position
    x += pos_start[0]
    y += pos_start[1]
    z += pos_start[2]

    return x, y, z
