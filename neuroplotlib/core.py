import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PolyCollection

try:
    import LFPy
except:
    raise ImportError("'LFPy' not installed. Install it with 'pip install LFPy'")

try:
    import neuron
except:
    raise ImportError("'neuron' not installed. Install it from https://www.neuron.yale.edu/neuron/download")

from .utils import get_polygons_for_cylinder


def plot_detailed_neuron(cell=None, morphology=None, position=None, rotation=None, plane='yz', ax=None,
                         alpha=0.8, color='gray', exclude_sections=[], xlim=None, ylim=None, labelsize=15,
                         **clr_kwargs):
    '''
    Plots detailed morphology of neuron using pt3d info.

    Parameters
    ----------
    cell: LFPy.Cell
        The cell object to be plotted
    morphology: str
        The path to a morphology ('.asc', '.swc', '.hoc') in alternative to the 'cell' object
    position: np.array
    rotation: np.array
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
    if cell is None:
        assert morphology is not None, "Provide 'cell' (LFPy.Cell) or a morphology file ('.hoc', '.asc', '.swc')"
        cell = LFPy.Cell(morphology=morphology, pt3d=True)
    elif type(cell) is not LFPy.TemplateCell and type(cell) is not LFPy.Cell:
        raise AttributeError('Either a Cell object or the cell name and location should be passed as parameters')

    if position is not None:
        if len(position) != 3:
            print('Input a single position at a time')
        else:
            original_position = cell.somapos
            cell.set_pos(position[0], position[1], position[2])

    if rotation is not None:
        if len(rotation) != 3:
            print('Input a single rotation at a time')
        else:
            cell.set_rotation(rotation[0], rotation[1], rotation[2])

    if ax is None:
        fig = plt.figure()
        if plane is not '3d':
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')

    sec_names = []
    for sec in cell.allseclist:
        sec_name = sec.name()
        # take care of templates
        if '.' in sec_name:
            sec_name = sec_name.split('.')[1]
        # take care of multiple sections in the same neuronal part
        if '[' in sec_name:
            sec_name = sec_name.split('[')[0]
        sec_names.append(sec_name)
    parts = np.unique(sec_names)

    idxs = {}
    idxs_3d = {}
    colors = {}

    for part in parts:
        idxs[part] = []
        idxs_3d[part] = []

    # assign idxs to neuron parts
    for idx in cell.get_idx():
        for part in parts:
            secname = cell.get_idx_name(idx)[1]
            if part in secname:
                idxs[part].append(idx)
                break

    # assign 3d idxs to neuron parts
    for i in range(len(cell.x3d)):
        mid = len(cell.x3d[i]) // 2
        idx_3d = cell.get_closest_idx(cell.x3d[i][mid], cell.y3d[i][mid], cell.z3d[i][mid])
        for part in parts:
            if idx_3d in idxs[part]:
                idxs_3d[part].append(i)
                break

    # get colors
    for part in parts:
        if f"color_{part}" in clr_kwargs.keys():
            colors[part] = clr_kwargs[f"color_{part}"]
        else:
            colors[part] = color

    zips = []
    if plane is '3d':
        for part in parts:
            if part not in exclude_sections:
                _plot_3d_neurites(cell, ax, colors[part], alpha, idxs=idxs[part], pt3d=True)
        gmax = np.max([np.max(np.abs(cell.xmid)), np.abs(np.max(cell.ymid)), np.abs(np.max(cell.zmid))])
        ax.set_xlim3d(-gmax, gmax)
        ax.set_ylim3d(-gmax, gmax)
        ax.set_zlim3d(-gmax, gmax)
    else:
        if plane is 'yz' or plane is 'zy':
            for y, z in cell.get_pt3d_polygons(projection=('y', 'z')):
                zips.append(zip(y, z))
        elif plane is 'xz' or plane is 'zx':
            for x, z in cell.get_pt3d_polygons(projection=('x', 'z')):
                zips.append(zip(x, z))

        elif plane is 'xy' or plane is 'yx':
            for x, y in cell.get_pt3d_polygons(projection=('x', 'y')):
                zips.append(zip(x, y))

        for part in parts:
            if part not in exclude_sections:
                polygons = [list(zips[i]) for i in idxs_3d[part]]
                polycol = PolyCollection(polygons,
                                         edgecolors='none',
                                         facecolors=colors[part],
                                         alpha=alpha)
                ax.add_collection(polycol)

        if plane is 'xy' or plane is 'yx':
            ax.set_xlabel('x ($\mu$m)', fontsize=labelsize)
            ax.set_ylabel('y ($\mu$m)', fontsize=labelsize)
        elif plane is 'yz' or plane is 'zy':
            ax.set_xlabel('y ($\mu$m)', fontsize=labelsize)
            ax.set_ylabel('z ($\mu$m)', fontsize=labelsize)
        elif plane is 'xz' or plane is 'zx':
            ax.set_xlabel('x ($\mu$m)', fontsize=labelsize)
            ax.set_ylabel('z ($\mu$m)', fontsize=labelsize)

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if not xlim and not ylim:
            ax.axis('equal')

    # revert rotation and position
    if rotation is not None:
        if len(rotation) != 3:
            print('Input a single rotation at a time')
        else:
            cell.set_rotation(0, 0, -rotation[2])
            cell.set_rotation(0, -rotation[1], 0)
            cell.set_rotation(-rotation[0], 0, 0)

    if position is not None:
        if len(position) != 3:
            print('Input a single position at a time')
        else:
            cell.set_position(original_position[0], original_position[1], original_position[2])

    return ax


def plot_neuron(cell=None, morphology=None, position=None, rotation=None, plane='yz',
                fig=None, ax=None, projections3d=False, alpha=0.8, color='k', exclude_sections=[],
                xlim=None, ylim=None, labelsize=15, lw=1, **clr_kwargs):
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
    if cell is None:
        assert morphology is not None, "Provide 'cell' (LFPy.Cell) or a morphology file ('.hoc', '.asc', '.swc')"
        cell = LFPy.Cell(morphology=morphology, pt3d=True)
    elif type(cell) is not LFPy.TemplateCell and type(cell) is not LFPy.Cell:
        raise AttributeError('Either a Cell object or the cell name and location should be passed as parameters')

    if position is not None:
        if len(position) != 3:
            print('Input a single position at a time')
        else:
            original_position = cell.somapos
            cell.set_pos(position[0], position[1], position[2])
    if rotation is not None:
        if len(rotation) != 3:
            print('Input a single rotation at a time')
        else:
            cell.set_rotation(rotation[0], rotation[1], rotation[2])

    sec_names = []
    for sec in cell.allseclist:
        sec_name = sec.name()
        # take care of templates
        if '.' in sec_name:
            sec_name = sec_name.split('.')[1]
        # take care of multiple sections in the same neuronal part
        if '[' in sec_name:
            sec_name = sec_name.split('[')[0]
        sec_names.append(sec_name)
    parts = np.unique(sec_names)

    idxs = {}
    idxs_3d = {}
    colors = {}

    for part in parts:
        idxs[part] = []
        idxs_3d[part] = []

    # assign idxs to neuron parts
    for idx in cell.get_idx():
        for part in parts:
            secname = cell.get_idx_name(idx)[1]
            if part in secname:
                idxs[part].append(idx)
                break

    # get colors
    for part in parts:
        if f"color_{part}" in clr_kwargs.keys():
            colors[part] = clr_kwargs[f"color_{part}"]
        else:
            colors[part] = color

    if projections3d:
        if fig is None:
            fig = plt.figure()
        yz = fig.add_subplot(221, aspect=1)
        xy = fig.add_subplot(222, aspect=1)
        xz = fig.add_subplot(223, aspect=1)
        ax_3d = fig.add_subplot(224, projection='3d')

        for part in parts:
            if part not in exclude_sections:
                if 'soma' in part:
                    _plot_soma_ellipse(cell, idxs[part], 'xy', xy, color_soma=colors[part], alpha=alpha)
                    _plot_soma_ellipse(cell, idxs[part], 'yz', yz, color_soma=colors[part], alpha=alpha)
                    _plot_soma_ellipse(cell, idxs[part], 'xz', xz, color_soma=colors[part], alpha=alpha)
                    _plot_3d_neurites(cell, ax_3d, color=colors[part], alpha=alpha, idxs=idxs[part], pt3d=True)
                else:
                    for idx in idxs[part]:
                        xy.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]],
                                color=colors[part])
                        yz.plot([cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]],
                                color=colors[part])
                        xz.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]],
                                color=colors[part])
                        ax_3d.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]],
                                   [cell.zstart[idx], cell.zend[idx]], color=colors[part])

        yz.set_xlabel('y ($\mu$m)')
        yz.set_ylabel('z ($\mu$m)')
        xy.set_xlabel('x ($\mu$m)')
        xy.set_ylabel('y ($\mu$m)')
        xz.set_xlabel('x ($\mu$m)')
        xz.set_ylabel('z ($\mu$m)')
        ax_3d.set_xlabel('x ($\mu$m)')
        ax_3d.set_ylabel('y ($\mu$m)')
        ax_3d.set_zlabel('z ($\mu$m)')

        return fig
    else:
        if ax is None:
            fig = plt.figure()
            if plane is not '3d':
                ax = fig.add_subplot(111, aspect=1)
            else:
                ax = fig.add_subplot(111, projection='3d')

        if plane is not '3d':
            for part in parts:
                if part not in exclude_sections:
                    if 'soma' in part:
                        _plot_soma_ellipse(cell, idxs[part], plane, ax, color_soma=colors[part], alpha=alpha)
                    else:
                        for idx in idxs[part]:
                            if plane is 'xy' or plane is 'yx':
                                ax.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]],
                                        color=colors[part], lw=lw,
                                        alpha=alpha, zorder=3)
                            elif plane is 'yz' or plane is 'zy':
                                ax.plot([cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]],
                                        color=colors[part], lw=lw,
                                        alpha=alpha, zorder=3)
                            elif plane is 'xz' or plane is 'zx':
                                ax.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]],
                                        color=colors[part], lw=lw,
                                        alpha=alpha, zorder=3)
            if plane is 'xy' or plane is 'yx':
                ax.set_xlabel('x ($\mu$m)', fontsize=labelsize)
                ax.set_ylabel('y ($\mu$m)', fontsize=labelsize)
            elif plane is 'yz' or plane is 'zy':
                ax.set_xlabel('y ($\mu$m)', fontsize=labelsize)
                ax.set_ylabel('z ($\mu$m)', fontsize=labelsize)
            elif plane is 'xz' or plane is 'zx':
                ax.set_xlabel('x ($\mu$m)', fontsize=labelsize)
                ax.set_ylabel('z ($\mu$m)', fontsize=labelsize)

            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
            if not xlim and not ylim:
                ax.axis('equal')
        elif plane is '3d':
            for part in parts:
                if part not in exclude_sections:
                    if 'soma' in part:
                        _plot_3d_neurites(cell, ax, colors[part], alpha, idxs=idxs[part], pt3d=True)
                    else:
                        _plot_3d_neurites(cell, ax, colors[part], alpha, idxs=idxs[part], pt3d=False)
            ax.set_xlabel('x ($\mu$m)')
            ax.set_ylabel('y ($\mu$m)')
            ax.set_zlabel('z ($\mu$m)')
            ax.set_xlim3d(np.min(cell.xmid), np.max(cell.xmid))
            ax.set_ylim3d(np.min(cell.ymid), np.max(cell.ymid))
            ax.set_zlim3d(np.min(cell.zmid), np.max(cell.zmid))
        else:
            raise ValueError("Invalid 'plane'. It can be 'xy', 'yz', 'xz', or '3d'")

    # revert rotation and position
    if rotation is not None:
        if len(rotation) != 3:
            print('Input a single rotation at a time')
        else:
            cell.set_rotation(0, 0, -rotation[2])
            cell.set_rotation(0, -rotation[1], 0)
            cell.set_rotation(-rotation[0], 0, 0)

    if position is not None:
        if len(position) != 3:
            print('Input a single position at a time')
        else:
            cell.set_position(original_position[0], original_position[1], original_position[2])

    return ax


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
                                       alpha=alpha, lw=0.)

    for crt_poly3d in poly3d:
        ax.add_collection3d(crt_poly3d)

    if xlim:
        ax.set_xlim3d(xlim)
    if ylim:
        ax.set_xlim3d(ylim)
    if zlim:
        ax.set_xlim3d(zlim)

    return ax


def _plot_soma_ellipse(cell, idx_soma, plane, ax, color_soma, alpha=1):
    if isinstance(idx_soma, list):
        idx = idx_soma[0]
    else:
        idx = idx_soma
    width = cell.diam[idx]
    if plane == 'xy':
        height = np.sqrt((cell.zend[idx] - cell.zstart[idx]) ** 2 + (cell.yend[idx] - cell.ystart[idx]) ** 2)
        if cell.xend[idx] - cell.xstart[idx] != 0:
            angle = np.rad2deg((cell.yend[idx] - cell.ystart[idx]) / (cell.xend[idx] - cell.xstart[idx]))
        else:
            angle = 90
        xy = [cell.somapos[idx], cell.somapos[1]]
    elif plane == 'yz':
        height = np.sqrt((cell.yend[idx] - cell.ystart[idx]) ** 2 + (cell.zend[idx] - cell.zstart[idx]) ** 2)
        if cell.yend[idx] - cell.ystart[idx] != 0:
            angle = np.rad2deg((cell.zend[idx] - cell.zstart[idx]) / (cell.yend[idx] - cell.ystart[idx]))
        else:
            angle = 90
        xy = [cell.somapos[1], cell.somapos[2]]
    elif plane == 'xz':
        height = np.sqrt((cell.xend[idx] - cell.xstart[idx]) ** 2 + (cell.zend[idx] - cell.zstart[idx]) ** 2)
        if cell.xend[idx] - cell.xstart[idx] != 0:
            angle = np.rad2deg((cell.zend[idx] - cell.zstart[idx]) / (cell.xend[idx] - cell.xstart[idx]))
        else:
            angle = 90
        xy = [cell.somapos[0], cell.somapos[2]]

    e = Ellipse(xy=xy,
                width=width,
                height=height,
                angle=angle,
                color=color_soma,
                zorder=10,
                alpha=alpha)
    ax.add_artist(e)


def _plot_3d_neurites(cell, ax, color, alpha, idxs=None, pt3d=False):
    if idxs is None:
        idxs = cell.get_idx()
    if pt3d and cell.pt3d:
        for ii in range(len(cell.x3d)):
            for jj in range(len(cell.x3d[ii]) - 1):
                closest_idx = cell.get_closest_idx(cell.x3d[ii][jj], cell.y3d[ii][jj], cell.z3d[ii][jj])
                if closest_idx in idxs:
                    init = np.array([cell.x3d[ii][jj], cell.y3d[ii][jj], cell.z3d[ii][jj]])
                    end = np.array([cell.x3d[ii][jj + 1], cell.y3d[ii][jj + 1], cell.z3d[ii][jj + 1]])
                    dir_seg = (end - init) / np.linalg.norm(end - init)
                    len_seg = np.linalg.norm(end - init)
                    n_points = 10.
                    neur_poly3d = get_polygons_for_cylinder(init,
                                                            direction=dir_seg,
                                                            length=len_seg,
                                                            radius=cell.diam3d[ii][jj] / 2,
                                                            n_points=n_points,
                                                            facecolor=color,
                                                            edgecolor=color,
                                                            lw=0.5,
                                                            alpha=alpha)
                    for crt_poly3d in neur_poly3d:
                        ax.add_collection3d(crt_poly3d)
    else:
        for idx in idxs:
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
                                                    facecolor=color,
                                                    alpha=alpha)
    for crt_poly3d in neur_poly3d:
        ax.add_collection3d(crt_poly3d)
