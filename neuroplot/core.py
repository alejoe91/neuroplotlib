#!/usr/bin/env python
''' Clean up!!! '''


import os
from os.path import join
import numpy as np
import math

import matplotlib
import pylab as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
import MEAutility as MEA
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from matplotlib.collections import PolyCollection
from matplotlib import colors as mpl_colors
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# TODO update plots to cope with different MEAs


def rotation_matrix(d):
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
    skew = np.array([[    0,  d[2],  -d[1]],
                  [-d[2],     0,  d[0]],
                  [d[1], -d[0],    0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M

def make_patch_3d(pathpatch, rot_axis, angle, z=0):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color

    verts = path.vertices #Get the vertices in 2D

    M = rotation_matrix2(rot_axis, angle) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])

def pathpatch_2d_to_3d(pathpatch, z = 0, normal = 'z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0,0,0), index)

    normal /= np.linalg.norm(normal) #Make sure the vector is normalised

    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color

    verts = path.vertices #Get the vertices in 2D
    raise Exception()

    d = np.cross(normal, (0, 0, 1)) #Obtain the rotation vector
    M = rotation_matrix(d) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])


def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta


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

def rotation_matrix2(axis, theta):
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


def _cylinder(pos_start, direction, length, radius, n_points, flatten_along_zaxis=False):
    '''

    Parameters
    ----------
    pos_start
    direction
    length
    radius
    n_points
    flatten_along_zaxis

    Returns
    -------

    '''
    alpha = np.array([0., length])

    theta_ring = np.linspace(0., np.pi * 2., n_points)
    r = radius

    x = np.zeros((theta_ring.size * alpha.size))
    y = np.zeros((theta_ring.size * alpha.size))
    z = np.zeros((theta_ring.size * alpha.size))

    for idx_alpha, crt_alpha in enumerate(alpha):
        x[idx_alpha * theta_ring.size:
        (idx_alpha + 1) * theta_ring.size] = \
            r * np.cos(theta_ring)
        y[idx_alpha * theta_ring.size:
        (idx_alpha + 1) * theta_ring.size] = \
            r * np.sin(theta_ring)
        z[idx_alpha * theta_ring.size:
        (idx_alpha + 1) * theta_ring.size] = \
            crt_alpha * np.ones(theta_ring.size)

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    z = np.atleast_2d(z)

    d = direction

    # rot1, phi
    r_1 = np.array([0., 1., 0.])
    # rot2, theta
    r_2 = np.array([0., 0., 1.])

    # fix negative angles
    if d[0] == 0:
        theta = -np.sign(d[1])*np.pi / 2.
    else:
        if d[0] > 0:
            theta = -np.arctan(d[1] / d[0])
        else:
            theta = np.pi - np.arctan(d[1] / d[0])

    rho = np.sqrt((d[0] ** 2 + d[1] ** 2))

    if rho == 0:
        phi = 0.
    else:
        phi = -(np.pi / 2. - np.arctan(d[2] / rho))

    # print 'phi: ', np.rad2deg(phi)

    rot1_m = _rotation_matrix(r_1, phi)
    rot2_m = _rotation_matrix(r_2, theta)

    for idx, (crt_x, crt_y, crt_z) in enumerate(zip(x[0], y[0], z[0])):
        crt_v = np.array([crt_x, crt_y, crt_z])
        crt_v = np.dot(crt_v, rot1_m)
        crt_v = np.dot(crt_v, rot2_m)
        x[0][idx] = crt_v[0]
        y[0][idx] = crt_v[1]
        z[0][idx] = crt_v[2]

    x += pos_start[0]
    y += pos_start[1]
    z += pos_start[2]
    if flatten_along_zaxis is True:
        z = np.abs(z)
        z *= 0.00000000001
    return x, y, z


def get_polygons_for_cylinder(pos_start, direction, length, radius, n_points, facecolor='b', edgecolor='k', alpha=1.,
                              lw = 0., flatten_along_zaxis=False):
    '''

    Parameters
    ----------
    pos_start
    direction
    length
    radius
    n_points
    facecolor
    edgecolor
    alpha
    lw
    flatten_along_zaxis

    Returns
    -------

    '''
    x, y, z = _cylinder(pos_start,
                        direction,
                        length,
                        radius,
                        n_points,
                        flatten_along_zaxis)

    alpha_tup = alpha,
    edge_col = mpl_colors.to_rgb(edgecolor) + alpha_tup
    face_col = mpl_colors.to_rgb(facecolor) + alpha_tup

    theta_ring = np.linspace(0., np.pi * 2., n_points)
    verts_hull = []
    for idx_theta, crt_theta in enumerate(theta_ring):
        if idx_theta <= theta_ring.size - 2:
            x_verts = [x[0][idx_theta],
                       x[0][idx_theta + 1],
                       x[0][idx_theta + 1 + theta_ring.size],
                       x[0][idx_theta + theta_ring.size]]
            y_verts = [y[0][idx_theta],
                       y[0][idx_theta + 1],
                       y[0][idx_theta + 1 + theta_ring.size],
                       y[0][idx_theta + theta_ring.size]]
            z_verts = [z[0][idx_theta],
                       z[0][idx_theta + 1],
                       z[0][idx_theta + 1 + theta_ring.size],
                       z[0][idx_theta + theta_ring.size]]
            verts_hull.append([zip(x_verts, y_verts, z_verts)])

    poly3d_hull = []
    for crt_vert in verts_hull:
        cyl = Poly3DCollection(crt_vert, linewidths=lw)
        cyl.set_facecolor(face_col)
        cyl.set_edgecolor(edge_col)

        poly3d_hull.append(cyl)

    # draw lower lid
    x_verts = x[0][0:theta_ring.size - 1]
    y_verts = y[0][0:theta_ring.size - 1]
    z_verts = z[0][0:theta_ring.size - 1]
    verts_lowerlid = [zip(x_verts, y_verts, z_verts)]
    poly3ed_lowerlid = Poly3DCollection(verts_lowerlid, linewidths=lw, zorder=1)
    poly3ed_lowerlid.set_facecolor(face_col)
    poly3ed_lowerlid.set_edgecolor(edge_col)

    # draw upper lid
    x_verts = x[0][theta_ring.size:theta_ring.size * 2 - 1]
    y_verts = y[0][theta_ring.size:theta_ring.size * 2 - 1]
    z_verts = z[0][theta_ring.size:theta_ring.size * 2 - 1]
    verts_upperlid = [zip(x_verts, y_verts, z_verts)]
    poly3ed_upperlid = Poly3DCollection(verts_upperlid, linewidths=lw, zorder=1)
    poly3ed_upperlid.set_facecolor(face_col)
    poly3ed_upperlid.set_edgecolor(edge_col)

    return_col = poly3d_hull
    return_col.append(poly3ed_lowerlid)
    return_col.append(poly3ed_upperlid)
    return return_col


def plot_detailed_neuron(cell=None, cell_name=None, cell_folder=None, pos=None, rot=None, plane=None, ax=None,
                         bounds=None, alpha=None, color=None, c_axon=None, c_dend=None, c_soma=None, plot_axon=True,
                         xlim=None, ylim=None):
    '''

    Parameters
    ----------
    cell
    cell_name
    cell_folder
    pos
    rot
    plane
    ax
    bounds
    alpha
    color
    c_axon
    c_dend
    c_soma
    plot_axon
    xlim
    ylim

    Returns
    -------

    '''
    import LFPy

    if cell is None:
        if cell_name is None or cell_folder is None:
            raise AttributeError('Either a Cell object or the cell name and location should be passed as parameters')
        folder = join(cell_folder, cell_name)
        cwd = os.getcwd()
        os.chdir(folder)
        morphologyfile = os.listdir('morphology')[0]  # glob('morphology\\*')[0]
        cell = LFPy.Cell(morphology=join('morphology', morphologyfile),
                         pt3d=True)
        os.chdir(cwd)
    elif type(cell) is not LFPy.TemplateCell and type(cell) is not LFPy.Cell:
        raise AttributeError('Either a Cell object or the cell name and location should be passed as parameters')

    if pos is not None:
        if len(pos) != 3:
            print 'Input a single posiion at a time'
        else:
            cell.set_pos(pos[0], pos[1], pos[2])
    if rot is not None:
        if len(rot) != 3:
            print 'Input a single posiion at a time'
        else:
            cell.set_rotation(rot[0], rot[1], rot[2])

    if alpha:
        alp = alpha
    else:
        alp = 0.8
    if color:
        col = color
    else:
        col = 'gray'
    if c_axon:
        col_ax = c_axon
    else:
        col_ax = col
    if c_dend:
        col_dend = c_dend
    else:
        col_dend = col
    if c_soma:
        col_soma = c_soma
    else:
        col_soma = col
    if ax is None:
        fig = plt.figure()
        if plane is not '3d':
            axis = fig.add_subplot(111)
        else:
            axis = fig.add_subplot(111, projection='3d')
    else:
        axis = ax

    idx_ax = cell.get_idx('axon')
    idx_dend = cell.get_idx('dend')
    idx_apic = cell.get_idx('apic')
    idx_soma = cell.get_idx('soma')

    idx_ax3d = []
    idx_dend3d = []
    idx_apic3d = []
    idx_soma3d = []

    for i in range(len(cell.x3d)):
        mid = len(cell.x3d[i]) // 2
        idx = cell.get_closest_idx(cell.x3d[i][mid],cell.y3d[i][mid], cell.z3d[i][mid])
        if idx in idx_ax:
            idx_ax3d.append(i)
        elif idx in idx_dend:
            idx_dend3d.append(i)
        elif idx in idx_apic:
            idx_apic3d.append(i)
        elif idx in idx_soma:
            idx_soma3d.append(i)

    zips = []

    if plane is '3d':
        '''USE x3d, y3d, z3d and nested cycle...not better'''

        for ii in range(len(cell.x3d)):
            print 'seg: [', ii, '/', len(cell.x3d), ']'
            for jj in range(len(cell.x3d[ii])-1):
                init = np.array([cell.x3d[ii][jj], cell.y3d[ii][jj], cell.z3d[ii][jj]])
                end = np.array([cell.x3d[ii][jj+1], cell.y3d[ii][jj+1], cell.z3d[ii][jj+1]])
                dir_seg = (end -init) / np.linalg.norm(end-init)
                len_seg = np.linalg.norm(end-init)
                n_points = 5.
                neur_poly3d = get_polygons_for_cylinder(init,
                                                         direction=dir_seg,
                                                         length=len_seg,
                                                         radius=cell.diam3d[ii][jj]/2,
                                                         n_points=n_points,
                                                         facecolor=col,
                                                         alpha=alp,
                                                         flatten_along_zaxis=False)
                for crt_poly3d in neur_poly3d:
                    axis.add_collection3d(crt_poly3d)

        axis.set_xlim3d(np.min(cell.xmid), np.max(cell.xmid))
        axis.set_ylim3d(np.min(cell.ymid), np.max(cell.ymid))
        axis.set_zlim3d(np.min(cell.zmid), np.min(cell.zmid))

    elif plane is 'yz' or plane is 'zy':
        for y, z in cell.get_pt3d_polygons(projection=('y', 'z')):
            zips.append(zip(y, z))

    elif plane is 'xz' or plane is 'zx':
        for x, z in cell.get_pt3d_polygons(projection=('x', 'z')):
            zips.append(zip(x, z))

    elif plane is 'xy' or plane is 'yx':
        for x, y in cell.get_pt3d_polygons(projection=('x', 'y')):
            zips.append(zip(x, y))

    if plane is not '3d':
        axon_pol = [zips[i] for i in idx_ax3d]
        dend_pol = [zips[i] for i in idx_dend3d]
        apic_pol = [zips[i] for i in idx_apic3d]
        soma_pol = [zips[i] for i in idx_soma3d]
        if plot_axon:
            polycol_ax = PolyCollection(axon_pol,
                                     edgecolors='none',
                                     facecolors=col_ax,
                                     alpha=alp)
            axis.add_collection(polycol_ax)
        polycol_dend = PolyCollection(dend_pol,
                                 edgecolors='none',
                                 facecolors=col_dend,
                                 alpha=alp)
        polycol_apic = PolyCollection(apic_pol,
                                 edgecolors='none',
                                 facecolors=col_dend,
                                 alpha=alp)
        polycol_soma = PolyCollection(soma_pol,
                                 edgecolors='none',
                                 facecolors=col_soma,
                                 alpha=alp)

        axis.add_collection(polycol_dend)
        axis.add_collection(polycol_apic)
        axis.add_collection(polycol_soma)    

    if bounds is not None:
        if len(bounds) == 4:
            axis.axis('on')
            axis.set_xlim([bounds[0], bounds[1]])
            axis.set_ylim([bounds[2], bounds[3]])
            axis.axis('off')
    else:
        if plane is '3d':
            axis.set_xlim3d(np.min(cell.xmid), np.max(cell.xmid))
            axis.set_ylim3d(np.min(cell.ymid), np.max(cell.ymid))
            axis.set_zlim3d(np.min(cell.zmid), np.max(cell.zmid))
        else:
            if xlim:
                axis.set_xlim(xlim)
            if ylim:
                axis.set_ylim(ylim)
            if not xlim and not ylim:
                axis.axis('equal')

    return axis


def plot_neuron(cell=None, cell_name=None, cell_folder=None, pos=None, rot=None, bounds=None, plane=None,
                fig=None,ax=None, projections3d=False, alpha=None, color=None, condition=None,
                c_axon=None, c_dend=None, c_soma=None, plot_axon=True, plot_dend=True, plot_soma=True,
                xlim=None, ylim=None, somasize=30):
    '''

    Parameters
    ----------
    cell
    cell_name
    cell_folder
    pos
    rot
    bounds
    plane
    fig
    ax
    projections3d
    alpha
    color
    condition
    c_axon
    c_dend
    c_soma
    plot_axon
    plot_dend
    plot_soma
    xlim
    ylim
    somasize

    Returns
    -------

    '''
    import LFPy

    # in LFPy you can get all points of tue neuron
    # cell.get
    if cell is None:
        if cell_name is None or cell_folder is None:
            raise AttributeError('Either a Cell object or the cell name and location should be passed as parameters')
        folder = join(cell_folder, cell_name)
        cwd = os.getcwd()
        print folder
        os.chdir(folder)
        morphologyfile = os.listdir('morphology')[0]  # glob('morphology\\*')[0]
        print join('morphology', morphologyfile)
        cell = LFPy.Cell(morphology=join('morphology', morphologyfile))
        os.chdir(cwd)
    elif type(cell) is not LFPy.TemplateCell and type(cell) is not LFPy.Cell:
        raise AttributeError('Either a Cell object or the cell name and location should be passed as parameters')

    # cell = return_cell_shape(folder, cell_name)
    if pos is not None:
        if len(pos) != 3:
            print 'Input a single posiion at a time'
        else:
            cell.set_pos(pos[0], pos[1], pos[2])
    if rot is not None:
        if len(rot) != 3:
            print 'Input a single posiion at a time'
        else:
            cell.set_rotation(rot[0], rot[1], rot[2])

    if plane is None:
        plane = 'yz'
    if alpha:
        alp = alpha
    else:
        alp = 0.8
    if color:
        col = color
    else:
        col = 'gray'
    if c_axon:
        col_ax = c_axon
    else:
        col_ax = col
    if c_dend:
        col_dend = c_dend
    else:
        col_dend = col
    if c_soma:
        col_soma = c_soma
    else:
        col_soma = col

    idx_ax = cell.get_idx('axon')
    idx_dend = cell.get_idx('dend')
    idx_soma = cell.get_idx('soma')

    if projections3d:
        if fig is None:
            fig = plt.figure()
        yz = fig.add_subplot(221, aspect=1)
        xy = fig.add_subplot(222, aspect=1)
        xz = fig.add_subplot(223, aspect=1)
        threeD = fig.add_subplot(224, projection='3d')

        for idx in range(cell.totnsegs):
            if idx == cell.somaidx:
                yz.plot([cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]], 'r', lw=5)
            elif idx in idx_ax:
                yz.plot([cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]], 'b')
            else:
                yz.plot([cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]], 'k', lw=0.5)
        for idx in range(cell.totnsegs):
            if idx == cell.somaidx:
                xy.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], 'r', lw=5)
            elif idx in idx_ax:
                xy.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], 'b')
            else:
                xy.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], 'k', lw=0.5)
        for idx in range(cell.totnsegs):
            if idx == cell.somaidx:
                xz.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], 'r', lw=5)
            elif idx in idx_ax:
                xz.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], 'b')
            else:
                xz.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], 'k', lw=0.5)

        for idx in range(cell.totnsegs):
            if idx == cell.somaidx:
                threeD.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]],
                            [cell.zstart[idx], cell.zend[idx]], 'r', lw=5)
            elif idx in idx_ax:
                threeD.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]],
                            [cell.zstart[idx], cell.zend[idx]], 'b')
            else:
                threeD.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]],
                            [cell.zstart[idx], cell.zend[idx]], 'k', lw=0.5)

        yz.set_xlabel('y ($\mu$m)')
        yz.set_ylabel('z ($\mu$m)')
        xy.set_xlabel('x ($\mu$m)')
        xy.set_ylabel('y ($\mu$m)')
        xz.set_xlabel('x ($\mu$m)')
        xz.set_ylabel('z ($\mu$m)')
        threeD.set_xlabel('x ($\mu$m)')
        threeD.set_ylabel('y ($\mu$m)')
        threeD.set_zlabel('z ($\mu$m)')
        threeD.axis('equal')

        return fig 

    else:
        if ax:
            axis = ax
        else:
            fig = plt.figure()
            if plane is not '3d':
                axis = fig.add_subplot(111, aspect=1)
            else:
                axis = fig.add_subplot(111, projection='3d')

        if plane is 'xy':
            zips = []
            zips_ax = []
            zips_dend = []

            for x, y in cell.get_idx_polygons():
                zips.append(list(zip(x, y)))

            for idx in range(cell.totnsegs):
                if idx in idx_ax:
                    if plot_axon:
                        if method == 'lines':
                            axis.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]],
                                      color=col_ax, lw=lwid,
                                      alpha=alp, zorder=3)
                        elif method == 'polygons':
                            zips_ax.append(zips[idx])
                elif idx not in idx_soma:
                    if plot_dend:
                        if method == 'lines':
                            axis.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]],
                                      color=col_dend, lw=lwid,
                                      alpha=alp, zorder=2)
                        elif method == 'polygons':
                            zips_dend.append(zips[idx])

            if method == 'polygons':
                if len(zips_ax) > 1:
                    polycol = PolyCollection(zips_ax,
                                             edgecolors='none',
                                             facecolors=col_ax,
                                             alpha=alp,
                                             zorder=2)
                    axis.add_collection(polycol)
                if len(zips_dend) > 1:
                    polycol = PolyCollection(zips_dend,
                                             edgecolors='none',
                                             facecolors=col_dend,
                                             alpha=alp,
                                             zorder=2)
                    axis.add_collection(polycol)
            if plot_soma:
                height = np.sqrt((cell.xend[0] - cell.xstart[0]) ** 2 + (cell.yend[0] - cell.ystart[0]) ** 2)
                width = cell.diam[0]
                if (cell.xend[0] - cell.xstart[0]) != 0:
                    angle = np.rad2deg((cell.yend[0] - cell.ystart[0]) / (cell.xend[0] - cell.xstart[0]))
                else:
                    angle = 90

                e = Ellipse(xy=[cell.somapos[0], cell.somapos[1]],
                            width=width,
                            height=height,
                            angle=angle,
                            color=col_soma,
                            zorder=10,
                            alpha=alp)
                axis.add_artist(e)

            axis.set_xlabel('x ($\mu$m)', fontsize=labelsize)
            axis.set_ylabel('y ($\mu$m)', fontsize=labelsize)

        elif plane is 'yz':
            zips = []
            zips_ax = []
            zips_dend = []

            for y, z in cell.get_idx_polygons():
                zips.append(list(zip(y, z)))

            for idx in range(cell.totnsegs):
                if idx in idx_ax:
                    if plot_axon:
                        if method=='lines':
                            axis.plot([cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]],
                                      color=col_ax, lw=lwid,
                                      alpha=alp, zorder=3)
                        elif method=='polygons':
                            zips_ax.append(zips[idx])
                elif idx not in idx_soma:
                    if plot_dend:
                        if method=='lines':
                            axis.plot([cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]],
                                      color=col_dend, lw=lwid,
                                      alpha=alp, zorder=2)
                        elif method=='polygons':
                            zips_dend.append(zips[idx])

            if method=='polygons':
                if len(zips_ax) > 1:
                    polycol = PolyCollection(zips_ax,
                                             edgecolors='none',
                                             facecolors=col_ax,
                                             alpha=alp,
                                             zorder=2)
                    axis.add_collection(polycol)
                if len(zips_dend) > 1:
                    polycol = PolyCollection(zips_dend,
                                             edgecolors='none',
                                             facecolors=col_dend,
                                             alpha=alp,
                                             zorder=2)
                    axis.add_collection(polycol)

            if plot_soma:
                height = np.sqrt((cell.yend[0]-cell.ystart[0])**2 + (cell.zend[0]-cell.zstart[0])**2)
                width = cell.diam[0]
                if (cell.yend[0]-cell.ystart[0]) != 0:
                    angle = np.rad2deg((cell.zend[0]-cell.zstart[0])/(cell.yend[0]-cell.ystart[0]))
                else:
                    angle = 90

                e = Ellipse(xy=[cell.somapos[1], cell.somapos[2]],
                            width=width,
                            height=height,
                            angle=angle,
                            color=col_soma,
                            zorder=10,
                            alpha=alp)
                axis.add_artist(e)

            axis.set_xlabel('y ($\mu$m)', fontsize=labelsize)
            axis.set_ylabel('z ($\mu$m)', fontsize=labelsize)

        elif plane is 'xz':
            zips = []
            zips_ax = []
            zips_dend = []

            for x, z in cell.get_idx_polygons():
                zips.append(list(zip(x, z)))

            for idx in range(cell.totnsegs):
                if idx in idx_ax:
                    if plot_axon:
                        if method == 'lines':
                            axis.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]],
                                      color=col_ax, lw=lwid,
                                      alpha=alp, zorder=3)
                        elif method == 'polygons':
                            zips_ax.append(zips[idx])
                elif idx not in idx_soma:
                    if plot_dend:
                        if method == 'lines':
                            axis.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]],
                                      color=col_dend, lw=lwid,
                                      alpha=alp, zorder=2)
                        elif method == 'polygons':
                            zips_dend.append(zips[idx])

            if method == 'polygons':
                if len(zips_ax) > 1:
                    polycol = PolyCollection(zips_ax,
                                             edgecolors='none',
                                             facecolors=col_ax,
                                             alpha=alp,
                                             zorder=2)
                    axis.add_collection(polycol)
                if len(zips_dend) > 1:
                    polycol = PolyCollection(zips_dend,
                                             edgecolors='none',
                                             facecolors=col_dend,
                                             alpha=alp,
                                             zorder=2)
                    axis.add_collection(polycol)
            if plot_soma:
                height = np.sqrt((cell.xend[0] - cell.xstart[0]) ** 2 + (cell.zend[0] - cell.zstart[0]) ** 2)
                width = cell.diam[0]
                if (cell.xend[0] - cell.xstart[0]) != 0:
                    angle = np.rad2deg((cell.zend[0] - cell.zstart[0]) / (cell.xend[0] - cell.xstart[0]))
                else:
                    angle=90

                e = Ellipse(xy=[cell.somapos[0], cell.somapos[2]],
                            width=width,
                            height=height,
                            angle=angle,
                            color=col_soma,
                            zorder=10,
                            alpha=alp)
                axis.add_artist(e)

            axis.set_xlabel('x ($\mu$m)', fontsize=labelsize)
            axis.set_ylabel('z ($\mu$m)', fontsize=labelsize)

        elif plane is '3d':
            # ax = plt.gca()
            # ax_3d = plt.subplot(111, projection='3d')
            # ax.axis('off')
            ax_3d = axis
            for idx in range(cell.totnsegs):
                if idx == cell.somaidx:
                    for jj in range(len(cell.x3d[idx]) - 1):
                        init = np.array([cell.x3d[idx][jj], cell.y3d[idx][jj], cell.z3d[idx][jj]])
                        end = np.array([cell.x3d[idx][jj + 1], cell.y3d[idx][jj + 1], cell.z3d[idx][jj + 1]])
                        dir_seg = (end - init) / np.linalg.norm(end - init)
                        len_seg = np.linalg.norm(end - init)
                        n_points = 10.
                        neur_poly3d = get_polygons_for_cylinder(init,
                                                                 direction=dir_seg,
                                                                 length=len_seg,
                                                                 radius=cell.diam3d[idx][jj] / 2,
                                                                 n_points=n_points,
                                                                 facecolor=col_soma,
                                                                 edgecolor=col_soma,
                                                                 lw = 0.5,
                                                                 alpha=alp,
                                                                 flatten_along_zaxis=False)
                        for crt_poly3d in neur_poly3d:
                            ax_3d.add_collection3d(crt_poly3d)
                    # ax_3d.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]],
                    #            [cell.zstart[idx], cell.zend[idx]], 'r', lw=15)
                elif idx in idx_ax:
                    init = np.array([cell.xstart[idx], cell.ystart[idx], cell.zstart[idx]])
                    end = np.array([cell.xend[idx], cell.yend[idx], cell.zend[idx]])
                    dir_seg = (end - init) / np.linalg.norm(end - init)
                    len_seg = np.linalg.norm(end - init)
                    n_points = 10.
                    neur_poly3d = get_polygons_for_cylinder(init,
                                                            direction=dir_seg,
                                                            length=len_seg,
                                                            radius=cell.diam[idx] / 2,
                                                            n_points=n_points,
                                                            facecolor=col_ax,
                                                            alpha=alp,
                                                            flatten_along_zaxis=False)
                    for crt_poly3d in neur_poly3d:
                        ax_3d.add_collection3d(crt_poly3d)
                    # ax_3d.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]],
                    #            [cell.zstart[idx], cell.zend[idx]], 'gray', lw=3)
                else:
                    # Add limit on x axis
                    if condition is not None:
                        if eval(condition):
                            init = np.array([cell.xstart[idx], cell.ystart[idx], cell.zstart[idx]])
                            end = np.array([cell.xend[idx], cell.yend[idx], cell.zend[idx]])
                            dir_seg = (end - init) / np.linalg.norm(end - init)
                            len_seg = np.linalg.norm(end - init)
                            n_points = 10.
                            neur_poly3d = get_polygons_for_cylinder(init,
                                                                    direction=dir_seg,
                                                                    length=len_seg,
                                                                    radius=cell.diam[idx] / 2,
                                                                    n_points=n_points,
                                                                    facecolor=col_dend,
                                                                    alpha=alp,
                                                                    flatten_along_zaxis=False)
                            for crt_poly3d in neur_poly3d:
                                ax_3d.add_collection3d(crt_poly3d)
                    else:
                        init = np.array([cell.xstart[idx], cell.ystart[idx], cell.zstart[idx]])
                        end = np.array([cell.xend[idx], cell.yend[idx], cell.zend[idx]])
                        dir_seg = (end - init) / np.linalg.norm(end - init)
                        len_seg = np.linalg.norm(end - init)
                        n_points = 10.
                        neur_poly3d = get_polygons_for_cylinder(init,
                                                                direction=dir_seg,
                                                                length=len_seg,
                                                                radius=cell.diam[idx] / 2,
                                                                n_points=n_points,
                                                                facecolor=col_dend,
                                                                alpha=alp,
                                                                flatten_along_zaxis=False)
                        for crt_poly3d in neur_poly3d:
                            ax_3d.add_collection3d(crt_poly3d)

            ax_3d.set_xlabel('x ($\mu$m)')
            ax_3d.set_ylabel('y ($\mu$m)')
            ax_3d.set_zlabel('z ($\mu$m)')
            axis = ax_3d

        if bounds is not None:
            if len(bounds) == 4:
                axis.axis('on')
                axis.set_xlim([bounds[0], bounds[1]])
                axis.set_ylim([bounds[2], bounds[3]])
                axis.axis('off')
        else:
            if plane is '3d':
                axis.set_xlim3d(np.min(cell.xmid), np.max(cell.xmid))
                axis.set_ylim3d(np.min(cell.ymid), np.max(cell.ymid))
                axis.set_zlim3d(np.min(cell.zmid), np.max(cell.zmid))
                # axis.axis('equal')
            else:
                if xlim:
                    axis.set_xlim(xlim)
                if ylim:
                    axis.set_ylim(ylim)
                if not xlim and not ylim:
                    axis.axis('equal')

        if rot is not None:
            if len(rot) != 3:
                print 'Input a single posiion at a time'
            else:
                cell.set_rotation(0, 0, -rot[2])
                cell.set_rotation(0, -rot[1], 0)
                cell.set_rotation(-rot[0], 0, 0)

        #del cell
        # return axis

def get_templatename(f):
    '''
    Assess from hoc file the templatename being specified within

    Arguments
    ---------
    f : file, mode 'r'

    Returns
    -------
    templatename : str

    '''
    templatename = None
    f = file("template.hoc", 'r')
    for line in f.readlines():
        if 'begintemplate' in line.split():
            templatename = line.split()[-1]
            print 'template {} found!'.format(templatename)
            continue
    return templatename


def return_cell_shape(cell_name, cell_folder):
    '''

    Parameters
    ----------
    cell_name
    cell_folder

    Returns
    -------

    '''
    import LFPy

    cwd = os.getcwd()
    os.chdir(cell_folder)
    print cell_folder
    morphologyfile = os.listdir('morphology')[0]  # glob('morphology\\*')[0]
    cell = LFPy.Cell(morphology=join('morphology', morphologyfile), pt3d=True, delete_sections=False)
    os.chdir(cwd)

    return cell

def plot_probe(mea_pos, mea_pitch, shape='square', elec_dim=10, axis=None, xlim=None, ylim=None):
    '''

    Parameters
    ----------
    mea_pos
    mea_pitch
    shape
    elec_dim
    axis
    xlim
    ylim

    Returns
    -------

    '''
    from matplotlib.path import Path
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection

    if axis:
        ax = axis
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    n_elec = mea_pos.shape[0]

    y_pitch = mea_pitch[0]
    z_pitch = mea_pitch[1]


    elec_size = elec_dim / 2
    elec_size = (np.min([y_pitch,z_pitch]) - 0.3*np.min([y_pitch,z_pitch]))/2.
    elec_dim = (np.min([y_pitch,z_pitch]) - 0.3*np.min([y_pitch,z_pitch]))

    min_y = np.min(mea_pos[:,1])
    max_y = np.max(mea_pos[:,1])
    min_z = np.min(mea_pos[:,2])
    max_z = np.max(mea_pos[:,2])
    center_y = 0
    probe_height = 200
    probe_top = max_z + probe_height
    prob_bottom = min_z - probe_height
    prob_corner = min_z - 0.1*probe_height
    probe_left = min_y - 0.1*probe_height
    probe_right = max_y + 0.1*probe_height

    print min_y, max_y, min_z, max_z


    verts = [
        (min_y - 2*elec_dim, probe_top),  # left, bottom
        (min_y - 2*elec_dim, prob_corner),  # left, top
        (center_y, prob_bottom),  # right, top
        (max_y + 2*elec_dim, prob_corner),  # right, bottom
        (max_y + 2*elec_dim, probe_top),
        (min_y - 2 * elec_dim, max_z + 2 * elec_dim) # ignored
    ]

    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,
             ]

    path = Path(verts, codes)

    patch = patches.PathPatch(path, facecolor='green', edgecolor='k', lw=0.5, alpha=0.3)
    ax.add_patch(patch)

    if shape == 'square':
        for e in range(n_elec):
            elec = patches.Rectangle((mea_pos[e, 1] - elec_size, mea_pos[e, 2] - elec_size), elec_dim,  elec_dim,
                                     alpha=0.7, facecolor='orange', edgecolor=[0.3, 0.3, 0.3], lw=0.5)

            ax.add_patch(elec)
    elif shape == 'circle':
        for e in range(n_elec):
            elec = patches.Circle((mea_pos[e, 1], mea_pos[e, 2]), elec_size,
                                     alpha=0.7, facecolor='orange', edgecolor=[0.3, 0.3, 0.3], lw=0.5)

            ax.add_patch(elec)

    ax.set_xlim(probe_left - 5*elec_dim, probe_right + 5*elec_dim)
    ax.set_ylim(prob_bottom - 5*elec_dim, probe_top + 5*elec_dim)
    # ax.axis('equal')

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


def plot_probe_3d(mea_pos, rot_axis, theta, pos=[0, 0, 0], shape='square', alpha=.5,
                  elec_dim=15, probe_name=None, ax=None, xlim=None, ylim=None, zlim=None, top=1000):
    '''

    Parameters
    ----------
    mea_pos
    mea_pitch
    shape
    elec_dim
    axis
    xlim
    ylim

    Returns
    -------

    '''
    from matplotlib.patches import Circle

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    M = rotation_matrix2(rot_axis, theta)
    rot_pos = np.dot(M, mea_pos.T).T
    rot_pos += np.array(pos)

    normal = np.cross(rot_pos[1]-rot_pos[0], rot_pos[-1]-rot_pos[0])

    if probe_name is not None:
        if 'neuronexus' in probe_name.lower():
            for elec in rot_pos:
                p = Circle((0, 0), elec_dim/2., facecolor='orange', alpha=alpha)
                ax.add_patch(p)
                make_patch_3d(p, rot_axis, theta+np.pi/2.)
                pathpatch_translate(p, elec)

        tip_el_y = np.min(mea_pos[:, 2])
        bottom = tip_el_y - 62
        cz = 62 + np.sqrt(22**2 - 18**2) + 9*25
        top = top

        x_shank = [0, 0, 0, 0, 0, 0, 0]
        y_shank = [-57, -57, -31, 0, 31, 57, 57]
        z_shank = [bottom + top, bottom + cz, bottom + 62, bottom, bottom + 62, bottom + cz, bottom + top]

        shank_coord = np.array([x_shank, y_shank, z_shank])
        shank_coord_rot = np.dot(M, shank_coord)

        r = Poly3DCollection([np.transpose(shank_coord_rot)])
        # r.set_facecolor('green')
        alpha = (0.3,)
        mea_col = mpl_colors.to_rgb('g') + alpha
        edge_col = mpl_colors.to_rgb('k') + alpha
        r.set_edgecolor(edge_col)
        r.set_facecolor(mea_col)
        ax.add_collection3d(r)


    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if zlim:
        ax.set_zlim(zlim)

    return rot_pos


def plot_cylinder_3d(bottom, direction, length, radius, color='k', alpha=.5, ax=None,
                     xlim=None, ylim=None, zlim=None):
    '''

    Parameters
    ----------
    bottom
    direction
    color
    alpha
    ax
    xlim
    ylim
    zlim

    Returns
    -------

    '''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    poly3d = get_polygons_for_cylinder(bottom, direction, length, radius, n_points=100, facecolor=color, edgecolor='k',
                              alpha=alpha, lw=0., flatten_along_zaxis=False)

    for crt_poly3d in poly3d:
        ax.add_collection3d(crt_poly3d)

    if xlim:
        ax.set_xlim3d(xlim)
    if ylim:
        ax.set_xlim3d(ylim)
    if zlim:
        ax .set_xlim3d(zlim)

    return ax


# def plot_mea_recording(spikes, mea_pos, mea_pitch, color='k', points=False, lw=1):
#     '''
#
#     Parameters
#     ----------
#     spikes
#     mea_pos
#     mea_pitch
#     color
#     points
#     lw
#
#     Returns
#     -------
#
#     '''
#     number_electrode = mea_pos.shape[0]
#
#     y_pos = np.unique(mea_pos[:, 1])
#     z_pos = np.unique(mea_pos[:, 2])
#
#     y_norm = mea_pos[:, 1] - np.mean(mea_pos[:, 1])
#     z_norm = mea_pos[:, 2] - np.mean(mea_pos[:, 2])
#
#     z_col = mea_pos[np.where(mea_pos[:, 1]==y_pos[len(y_pos)//2])[0], 2]
#     y_row = mea_pos[np.where(mea_pos[:, 2]==z_pos[len(y_pos)//2])[0], 1]
#
#     N_z = len(z_col)
#     N_y = len(y_row)
#     y_pitch = mea_pitch[0]
#     z_pitch = mea_pitch[1]
#     # y_width = np.ptp(y_row)+y_pitch
#     # z_width = np.ptp(z_col)+z_pitch
#     y_width = np.ptp(y_norm) + y_pitch
#     z_width = np.ptp(z_norm) + z_pitch
#     yoffset = abs(np.min(y_norm)) + y_pitch / 2.
#     zoffset = abs(np.min(z_norm)) + z_pitch / 2.
#
#     # plot spikes on grid
#     fig = plt.figure()
#     for el in range(number_electrode):
#         # use add axes to adjust position and size
#         w = 0.9*y_pitch/y_width
#         h = 0.9*z_pitch/z_width
#         l = 0.05+(y_norm[el] - y_pitch/2. + yoffset) / y_width * 0.9
#         b = 0.05+(z_norm[el] - z_pitch/2. + zoffset) / z_width * 0.9
#
#         rect = l, b, w, h
#
#         ax = fig.add_axes(rect)
#         if len(spikes.shape) == 3:  # multiple
#             if points:
#                 ax.plot(np.transpose(spikes[:, el, :]), linestyle='-', marker='o', ms=2, lw=lw)
#             else:
#                 ax.plot(np.transpose(spikes[:, el, :]), lw=lw)
#         else:
#             if points:
#                 ax.plot(spikes[el, :], color=color, linestyle='-', marker='o', ms=2, lw=lw)
#             else:
#                 ax.plot(spikes[el, :], color=color, lw=lw)
#
#         ax.set_ylim([np.min(spikes), np.max(spikes)])
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.axis('off')
#
#     # return axis registered for neuron overlapping and mea bounds
#     axneur = fig.add_axes([0.05, 0.05, 0.9, 0.9])
#     axneur.axis('off')
#     bounds = [np.min(mea_pos[:, 1]), np.max(mea_pos[:, 1]), np.min(mea_pos[:, 2]), np.max(mea_pos[:, 2])]
#     return fig, axneur, bounds


def plot_mea_recording(spikes, mea_pos, mea_pitch, colors=None, points=False, lw=1, ax=None, spacing=None,
                       scalebar=False, time=None, dt=None, vscale=None):
    '''

    Parameters
    ----------
    spikes
    mea_pos
    mea_pitch
    color
    points
    lw

    Returns
    -------

    '''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, frameon=False)
        no_tight = False
    else:
        no_tight = True

    if spacing is None:
        spacing = 0.1*np.max(mea_pitch)

    # normalize to min peak
    if vscale is None:
        LFPmin = 1.5*   np.max(np.abs(spikes))
        spike_norm = spikes / LFPmin * mea_pitch[1]
    else:
        spike_norm = spikes / vscale * mea_pitch[1]

    if colors is None:
        if len(spikes.shape) > 2:
            colors = plt.rcParams['axes.color_cycle']
        else:
            colors='k'

    number_electrode = mea_pos.shape[0]
    for el in range(number_electrode):
        if len(spikes.shape) == 3:  # multiple
            if points:
                for sp_i, sp in enumerate(spike_norm):
                    if len(colors) >= len(spike_norm) and len(colors) > 1:
                        ax.plot(np.linspace(0, mea_pitch[0]-spacing, spikes.shape[2]) + mea_pos[el, 1],
                                np.transpose(sp[el, :]) +
                                mea_pos[el, 2], linestyle='-', marker='o', ms=2, lw=lw, color=colors[sp_i],
                                label='EAP '+ str(sp_i+1))
                    elif len(colors) == 1:
                        ax.plot(np.linspace(0, mea_pitch[0] - spacing, spikes.shape[2]) + mea_pos[el, 1],
                                np.transpose(sp[el, :]) +
                                mea_pos[el, 2], linestyle='-', marker='o', ms=2, lw=lw, color=colors,
                                label='EAP ' + str(sp_i + 1))
            else:
                for sp_i, sp in enumerate(spike_norm):
                    if len(colors) >= len(spike_norm) and len(colors) > 1:
                        ax.plot(np.linspace(0, mea_pitch[0]-spacing, spikes.shape[2]) + mea_pos[el, 1],
                                np.transpose(sp[el, :]) + mea_pos[el, 2], lw=lw, color=colors[sp_i],
                                label='EAP '+str(sp_i+1))
                    elif len(colors) == 1:
                        ax.plot(np.linspace(0, mea_pitch[0] - spacing, spikes.shape[2]) + mea_pos[el, 1],
                                np.transpose(sp[el, :]) + mea_pos[el, 2], lw=lw, color=colors,
                                label='EAP ' + str(sp_i + 1))

        else:
            if points:
                ax.plot(np.linspace(0, mea_pitch[0]-spacing, spikes.shape[1]) + mea_pos[el, 1], spike_norm[el, :]
                        + mea_pos[el, 2], color=colors, linestyle='-', marker='o', ms=2, lw=lw)
            else:
                ax.plot(np.linspace(0, mea_pitch[0]-spacing, spikes.shape[1]) + mea_pos[el, 1], spike_norm[el, :] +
                        mea_pos[el, 2], color=colors, lw=lw)

        # ax.set_ylim([np.min(spikes), np.max(spikes)])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    if scalebar:
        if dt is None and time is None:
            raise AttributeError('Pass either dt or time in the argument')
        else:
            shift = 0.1*spacing
            pos_h = [np.min(mea_pos[:, 1]), np.min(mea_pos[:, 2]) - 1.5*mea_pitch[1]]
            if vscale is None:
                length_h = mea_pitch[1] * LFPmin / (LFPmin // 10 * 10)
            else:
                length_h = mea_pitch[1]
            pos_w = [np.min(mea_pos[:, 1]), np.min(mea_pos[:, 2]) - 1.5*mea_pitch[1]]
            length_w = mea_pitch[0]/5.

            ax.plot([pos_h[0], pos_h[0]], [pos_h[1], pos_h[1] + length_h], color='k', lw=2)
            if vscale is None:
                ax.text(pos_h[0]+shift, pos_h[1] + length_h / 2., str(int(LFPmin // 10 * 10)) + ' $\mu$V')
            else:
                ax.text(pos_h[0]+shift, pos_h[1] + length_h / 2., str(int(vscale)) + ' $\mu$V')
            ax.plot([pos_w[0], pos_w[0]+length_w], [pos_w[1], pos_w[1]], color='k', lw=2)
            ax.text(pos_w[0]+shift, pos_w[1]-length_h/3., str(time/5) + ' ms')

    if not no_tight:
        fig.tight_layout()

    return ax


def plot_max_trace(spike, mea_dim=None, ax=None):
    '''

    Parameters
    ----------
    spike
    mea_dim
    axis

    Returns
    -------

    '''
    #  check if number of spike is 1
    if len(spike.shape) == 3:
        print 'Plot one spike at a time!'
        return
    else:
        if ax:
            ax = ax
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        # find max (shape is only 0 and 1!)
        max_idx = np.unravel_index(spike.argmax(), spike.shape)
        if (mea_dim[0] * mea_dim[1]) == spike.shape[0]:
            mea_values = spike[:, max_idx[1]].reshape((mea_dim[0],mea_dim[1]))
            im = ax.matshow(np.transpose(mea_values))
            # fig.colorbar(im)
        else:
            print 'MEA dimensions are wrong!'


def plot_min_trace(spike, mea_dim=None, ax=None, peak_image=None, cmap='jet', style='mat', origin='lower'):
    '''

    Parameters
    ----------
    spike
    mea_dim
    ax
    peak_image
    cmap
    style
    origin

    Returns
    -------

    '''
    #  check if number of spike is 1
    if len(spike.shape) == 3:
        raise AttributeError('Plot one spike at a time!')
    else:
        if ax:
            ax = ax
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        # find min (shape is only 0 and 1!)
        if peak_image is None:
            min_idx = np.unravel_index(spike.argmin(), spike.shape)
            if (mea_dim[0] * mea_dim[1]) == spike.shape[0]:
                mea_values = spike[:, min_idx[1]].reshape((mea_dim[0], mea_dim[1]))
                if style == 'mat':
                    im = ax.matshow(np.transpose(mea_values), cmap=cmap, origin=origin)
                else:
                    im = ax.imshow(np.transpose(mea_values), cmap=cmap, origin=origin)
            else:
                print 'MEA dimensions are wrong!'
        else:
            im = ax.matshow(np.transpose(peak_image))

        ax.axis('off')
        return ax, im


def plot_weight(weight, mea_dim=None, ax=None, cmap='viridis', style='mat', origin='lower'):
    '''

    Parameters
    ----------
    weight
    mea_dim
    axis
    cmap
    style
    origin

    Returns
    -------

    '''
    #  check if number of spike is 1
    if len(weight.shape) == 3:
        raise AttributeError('Plot one weight at a time!')
    else:
        if ax:
            ax = ax
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        # find min (shape is only 0 and 1!)
        if (mea_dim[0] * mea_dim[1]) == weight.shape[0]:
            mea_values = weight.reshape((mea_dim[0], mea_dim[1]))
            if style == 'mat':
                im = ax.matshow(np.transpose(mea_values), cmap=cmap, origin=origin)
            else:
                im = ax.imshow(np.transpose(mea_values), cmap=cmap, origin=origin)
        else:
            raise Exception('MEA dimensions are wrong!')

        ax.axis('off')
        return ax, im



def play_spike(spike, mea_dim, time=None, save=False, ax=None, file=None):
    '''

    Parameters
    ----------
    spike
    mea_dim
    time
    save
    ax
    file

    Returns
    -------

    '''
    #  check if number of spike is 1
    if len(spike.shape) == 3:
        print 'Plot one spike at a time!'
        return
    else:
        if time:
            inter = time
        else:
            inter = 20

    # if save:
    #     plt.switch_backend('agg')
    # else:
    #     plt.switch_backend('qt4agg')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    z_min = np.min(spike)
    z_max = np.max(spike)

    im0 = ax.imshow(np.zeros((mea_dim[0], mea_dim[1])), vmin=z_min, vmax=z_max)
    fig.colorbar(im0)
    ims = []

    if (mea_dim[0] * mea_dim[1]) == spike.shape[0]:
        for t in range(spike.shape[1]):
            ims.append([ax.imshow(np.transpose(spike[:, t].reshape((mea_dim[0], mea_dim[1]))),
                                   vmin=z_min, vmax=z_max)])

    im_ani = animation.ArtistAnimation(fig, ims, interval=inter, repeat_delay=2500, blit=True)

    if save:
        plt.switch_backend('agg')
        mywriter = animation.FFMpegWriter(fps=60)
        if file:
            im_ani.save(file, writer=mywriter)
        else:
            im_ani.save('spike.mp4', writer=mywriter)

    return im_ani


#### from plotting convention
def mark_subplots(axes, letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ', xpos=0.05, ypos=0.95, fs=50):

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

    if not type(axes) is list:
        axes = [axes]

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

def color_axes(axes, clr):
    if not type(axes) is list:
        axes = [axes]
    for ax in axes:
        ax.tick_params(axis='x', colors=clr)
        ax.tick_params(axis='y', colors=clr)
        for spine in ax.spines.values():
            spine.set_edgecolor(clr)

