import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PolyCollection
from distutils.version import StrictVersion

try:
    import LFPy
    if StrictVersion(LFPy.__version__) < "2.2.0":
        raise ImportError
except:
    raise ImportError("'LFPy' not installed or not updated (required version >=2.2). "
                      "Install it with 'pip install LFPy>=2.2'")

try:
    import neuron
except:
    raise ImportError("'neuron' not installed. Install it with 'pip install neuron'")

from .utils import get_polygons_for_cylinder


def plot_detailed_neuron(cell=None, morphology=None, plane='yz', position=None, rotation=None,
                         alpha=0.8, color='gray', exclude_sections=[], xlim=None, ylim=None, zlim=None,
                         labelsize=15, ax=None, **clr_kwargs):
    '''
    Plots detailed morphology of neuron using pt3d info.

    Parameters
    ----------
    cell: LFPy.Cell
        The cell object to be plotted
    morphology: str
        The path to a morphology ('.asc', '.swc', '.hoc') in alternative to the 'cell' object
    plane: str
        The plane to plot ('xy', 'yz', 'xz', '3d')
    position: np.array
        3d position to move the neuron
    rotation: np.array
        3d rotation for the neuron
    alpha: float
        Alpha value
    color: Matplotlib color
        The default color
    exclude_sections: list
        List of sections to exclude from plotting (they should be substrings of the NEURON sections -- e.g. axon, dend)
    xlim: tuple
        x limits (if None, they are automatically adjusted)
    ylim: tuple
        y limits (if None, they are automatically adjusted)
    zlim: tuple
        z limits (if None, they are automatically adjusted)
    labelsize: int
        Label size for axis labels
    ax: Matplotlib axis
        The axis to use
    **clr_kwargs: color keyword arguments. The are in the form of "color_*section*", where *section* is one of the
                  available sections (e.g. color_dend='r')

    Returns
    -------
    ax: Matplotlib axis
        The axis with the plotted neuron
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
        if plane != '3d':
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
            sec_name = cell.get_idx_name(idx)[1]
            if '.' in sec_name:
                sec_name = sec_name.split('.')[1]

            # take care of multiple sections in the same neuronal part
            if '[' in sec_name:
                sec_name = sec_name.split('[')[0]
            if part in sec_name:
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
    if plane == '3d':
        for part in parts:
            if part not in exclude_sections:
                _plot_3d_neurites(cell, ax, colors[part], alpha, idxs=idxs[part], pt3d=True)
        gmax = np.max([np.max(np.abs(cell.x.mean(axis=-1))),
                       np.abs(np.max(cell.y.mean(axis=-1))),
                       np.abs(np.max(cell.z.mean(axis=-1)))])

        if xlim is None:
            ax.set_xlim3d(-gmax, gmax)
        else:
            ax.set_xlim3d(xlim)
        if ylim is None:
            ax.set_ylim3d(-gmax, gmax)
        else:
            ax.set_ylim3d(ylim)
        if zlim is None:
            ax.set_zlim3d(-gmax, gmax)
        else:
            ax.set_zlim3d(zlim)

        ax.set_xlabel('x ($\mu$m)', fontsize=labelsize)
        ax.set_ylabel('y ($\mu$m)', fontsize=labelsize)
        ax.set_zlabel('z ($\mu$m)', fontsize=labelsize)
    else:
        if plane == 'yz' or plane == 'zy':
            for y, z in cell.get_pt3d_polygons(projection=('y', 'z')):
                zips.append(zip(y, z))
        elif plane == 'xz' or plane == 'zx':
            for x, z in cell.get_pt3d_polygons(projection=('x', 'z')):
                zips.append(zip(x, z))

        elif plane == 'xy' or plane == 'yx':
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

        if plane == 'xy' or plane == 'yx':
            ax.set_xlabel('x ($\mu$m)', fontsize=labelsize)
            ax.set_ylabel('y ($\mu$m)', fontsize=labelsize)
        elif plane == 'yz' or plane == 'zy':
            ax.set_xlabel('y ($\mu$m)', fontsize=labelsize)
            ax.set_ylabel('z ($\mu$m)', fontsize=labelsize)
        elif plane == 'xz' or plane == 'zx':
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
            cell.set_pos(original_position[0], original_position[1], original_position[2])

    return ax


def plot_neuron(cell=None, morphology=None, plane='yz', position=None, rotation=None,
                projections3d=False, alpha=0.8, color='gray', exclude_sections=[], xlim=None, ylim=None, zlim=None,
                labelsize=15, lw=1, ax=None, **clr_kwargs):
    '''
    Plots the morphology of a neuron (without pt3d info).

    Parameters
    ----------
    cell: LFPy.Cell
        The cell object to be plotted
    morphology: str
        The path to a morphology ('.asc', '.swc', '.hoc') in alternative to the 'cell' object
    plane: str
        The plane to plot ('xy', 'yz', 'xz', '3d')
    position: np.array
        3d position to move the neuron
    rotation: np.array
        3d rotation for the neuron
    projections3d: bool
        If True, a figure with 'xy', 'yz', 'xz', and '3d' planes is plotted
    alpha: float
        Alpha value
    color: Matplotlib color
        The default color
    exclude_sections: list
        List of sections to exclude from plotting (they should be substrings of the NEURON sections -- e.g. axon, dend)
    xlim: tuple
        x limits (if None, they are automatically adjusted)
    ylim: tuple
        y limits (if None, they are automatically adjusted)
    zlim: tuple
        z limits (if None, they are automatically adjusted)
    labelsize: int
        Label size for axis labels
    lw: float
        Line width for neuronal lines
    ax: Matplotlib axis
        The axis to use
    **clr_kwargs: color keyword arguments. The are in the form of "color_*section*", where *section* is one of the
                  available sections (e.g. color_dend='r')

    Returns
    -------
    ax: Matplotlib axis
        If projection3d is False, the axis with the plotted neuron
    fig, axes:  Matplotlib figure and list of axis
        If projection3d is True, the figure containing the projections and the list of axis (yz, xy, xz, 3d)
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
            sec_name = cell.get_idx_name(idx)[1]
            if '.' in sec_name:
                sec_name = sec_name.split('.')[1]

            # take care of multiple sections in the same neuronal part
            if '[' in sec_name:
                sec_name = sec_name.split('[')[0]
            if part in sec_name:
                idxs[part].append(idx)
                break

    # get colors
    for part in parts:
        if f"color_{part}" in clr_kwargs.keys():
            colors[part] = clr_kwargs[f"color_{part}"]
        else:
            colors[part] = color

    if projections3d:
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
                        xy.plot(cell.x[idx, :], cell.y[idx, :],
                                color=colors[part], alpha=alpha)
                        yz.plot(cell.y[idx, :], cell.z[idx, :],
                                color=colors[part], alpha=alpha)
                        xz.plot(cell.x[idx, :], cell.z[idx, :],
                                color=colors[part], alpha=alpha)
                        ax_3d.plot(cell.x[idx, :], cell.y[idx, :],
                                   cell.z[idx, :], color=colors[part], alpha=alpha)

        yz.set_xlabel('y ($\mu$m)')
        yz.set_ylabel('z ($\mu$m)')
        xy.set_xlabel('x ($\mu$m)')
        xy.set_ylabel('y ($\mu$m)')
        xz.set_xlabel('x ($\mu$m)')
        xz.set_ylabel('z ($\mu$m)')
        ax_3d.set_xlabel('x ($\mu$m)')
        ax_3d.set_ylabel('y ($\mu$m)')
        ax_3d.set_zlabel('z ($\mu$m)')

        if xlim is not None:
            xy.set_xlim(xlim)
            xz.set_xlim(xlim)
            ax_3d.set_xlim3d(xlim)
        if ylim is not None:
            xy.set_ylim(ylim)
            yz.set_xlim(ylim)
            ax_3d.set_ylim3d(ylim)
        if zlim is not None:
            xz.set_ylim(zlim)
            yz.set_ylim(zlim)
            ax_3d.set_zlim3d(zlim)

        return fig, [yz, xy, xz, ax_3d]
    else:
        if ax is None:
            fig = plt.figure()
            if plane != '3d':
                ax = fig.add_subplot(111, aspect=1)
            else:
                ax = fig.add_subplot(111, projection='3d')

        if plane != '3d':
            for part in parts:
                if part not in exclude_sections:
                    if 'soma' in part:
                        _plot_soma_ellipse(cell, idxs[part], plane, ax, color_soma=colors[part], alpha=alpha)
                    else:
                        for idx in idxs[part]:
                            if plane == 'xy' or plane == 'yx':
                                ax.plot(cell.x[idx, :], cell.y[idx, :],
                                        color=colors[part], lw=lw,
                                        alpha=alpha, zorder=3)
                            elif plane == 'yz' or plane == 'zy':
                                ax.plot(cell.y[idx, :], cell.z[idx, :],
                                        color=colors[part], lw=lw,
                                        alpha=alpha, zorder=3)
                            elif plane == 'xz' or plane == 'zx':
                                ax.plot(cell.x[idx, :], cell.z[idx, :],
                                        color=colors[part], lw=lw,
                                        alpha=alpha, zorder=3)

            if plane == 'xy' or plane == 'yx':
                ax.set_xlabel('x ($\mu$m)', fontsize=labelsize)
                ax.set_ylabel('y ($\mu$m)', fontsize=labelsize)
            elif plane == 'yz' or plane == 'zy':
                ax.set_xlabel('y ($\mu$m)', fontsize=labelsize)
                ax.set_ylabel('z ($\mu$m)', fontsize=labelsize)
            elif plane == 'xz' or plane == 'zx':
                ax.set_xlabel('x ($\mu$m)', fontsize=labelsize)
                ax.set_ylabel('z ($\mu$m)', fontsize=labelsize)

            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
            if not xlim and not ylim:
                ax.axis('equal')

        elif plane == '3d':
            for part in parts:
                if part not in exclude_sections:
                    if 'soma' in part:
                        _plot_3d_neurites(cell, ax, colors[part], alpha, idxs=idxs[part], pt3d=True)
                    else:
                        _plot_3d_neurites(cell, ax, colors[part], alpha, idxs=idxs[part], pt3d=False)
            ax.set_xlabel('x ($\mu$m)')
            ax.set_ylabel('y ($\mu$m)')
            ax.set_zlabel('z ($\mu$m)')
            ax.set_xlim3d(np.min(cell.x.mean(axis=-1)), np.max(cell.x.mean(axis=-1)))
            ax.set_ylim3d(np.min(cell.y.mean(axis=-1)), np.max(cell.y.mean(axis=-1)))
            ax.set_zlim3d(np.min(cell.z.mean(axis=-1)), np.max(cell.z.mean(axis=-1)))
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
            cell.set_pos(original_position[0], original_position[1], original_position[2])

    return ax


def plot_cylinder_3d(bottom, direction, length, radius, color='gray', alpha=.5,
                     xlim=None, ylim=None, zlim=None, ax=None):
    '''

    Parameters
    ----------
    bottom: np.array
        3d position of the bottom of the cylinder
    direction: np.array
        3d direction of the cylinder axis
    length: float
        Length of the cylinder
    radius: float
        Radius of the cylinder
    color: Matplotlib color
        The default color
    xlim: tuple
        x limits (if None, they are automatically adjusted)
    ylim: tuple
        y limits (if None, they are automatically adjusted)
    zlim: tuple
        z limits (if None, they are automatically adjusted)
    ax: Matplotlib axis
        The axis to use

    Returns
    -------
    ax: Matplotlib axis
        The axis with the plotted cylinder

    '''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    poly3d = get_polygons_for_cylinder(bottom, direction, length, radius, n_points=100, facecolor=color, edgecolor='k',
                                       alpha=alpha, lw=0.)

    for crt_poly3d in poly3d:
        ax.add_collection3d(crt_poly3d)
    top = bottom + length * np.array(direction)
    max_bottom = [np.max(np.abs(bottom[0])), np.abs(np.max(bottom[1])), np.abs(np.max(bottom[2]))]
    max_top = [np.max(np.abs(top[0])), np.abs(np.max(top[1])), np.abs(np.max(top[2]))]
    gmax = np.max([max_bottom, max_top])

    if xlim is None:
        ax.set_xlim3d(-gmax, gmax)
    else:
        ax.set_xlim3d(xlim)
    if ylim is None:
        ax.set_ylim3d(-gmax, gmax)
    else:
        ax.set_ylim3d(ylim)
    if zlim is None:
        ax.set_zlim3d(-gmax, gmax)
    else:
        ax.set_zlim3d(zlim)

    return ax


def _plot_soma_ellipse(cell, idx_soma, plane, ax, color_soma, alpha=1.,
                       as_circle=True):
    if isinstance(idx_soma, list):
        idx = idx_soma[0]
    else:
        idx = idx_soma
    width = cell.d[idx]
    if plane == 'xy':
        height = np.sqrt((np.diff(cell.z[idx, :])) ** 2 + (np.diff(cell.y[idx, :])) ** 2)
        if np.diff(cell.x[idx, :]) != 0:
            angle = np.rad2deg((np.diff(cell.y[idx, :])) / (np.diff(cell.x[idx, :])))
        else:
            angle = 90
        xy = [cell.somapos[idx], cell.somapos[1]]
    elif plane == 'yz':
        height = np.sqrt((np.diff(cell.y[idx, :])) ** 2 + (np.diff(cell.z[idx, :])) ** 2)
        if np.diff(cell.y[idx, :]) != 0:
            angle = np.rad2deg((np.diff(cell.z[idx, :])) / (np.diff(cell.y[idx, :])))
        else:
            angle = 90
        xy = [cell.somapos[1], cell.somapos[2]]
    elif plane == 'xz':
        height = np.sqrt((np.diff(cell.x[idx, :])) ** 2 + (np.diff(cell.z[idx, :])) ** 2)
        if np.diff(cell.x[idx, :]) != 0:
            angle = np.rad2deg((np.diff(cell.z[idx, :])) / (np.diff(cell.x[idx, :])))
        else:
            angle = 90
        xy = [cell.somapos[0], cell.somapos[2]]

    if as_circle:
        height = width
        angle = 0

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
        for idx in range(len(cell.x3d)):
            for jj in range(len(cell.x3d[idx]) - 1):
                midpoint = [cell.x3d[idx][jj] + cell.x3d[idx][jj + 1] - cell.x3d[idx][jj],
                            cell.y3d[idx][jj] + cell.y3d[idx][jj + 1] - cell.y3d[idx][jj],
                            cell.z3d[idx][jj] + cell.z3d[idx][jj + 1] - cell.z3d[idx][jj]]
                closest_idx = cell.get_closest_idx(midpoint[0], midpoint[1], midpoint[2])
                if closest_idx in idxs:
                    init = np.array([cell.x3d[idx][jj], cell.y3d[idx][jj], cell.z3d[idx][jj]])
                    end = np.array([cell.x3d[idx][jj + 1], cell.y3d[idx][jj + 1], cell.z3d[idx][jj + 1]])
                    len_seg = np.linalg.norm(end - init)
                    if len_seg > 0:
                        dir_seg = (end - init) / len_seg
                        n_points = 10
                        neur_poly3d = get_polygons_for_cylinder(init,
                                                                direction=dir_seg,
                                                                length=len_seg,
                                                                radius=cell.diam3d[idx][jj] / 2,
                                                                n_points=n_points,
                                                                facecolor=color,
                                                                edgecolor=color,
                                                                lw=0.5,
                                                                alpha=alpha)
                        for crt_poly3d in neur_poly3d:
                            ax.add_collection3d(crt_poly3d)
    else:
        for idx in idxs:
            init = np.array([cell.x[idx, 0], cell.y[idx, 0], cell.z[idx, 0]])
            end = np.array([cell.x[idx, -1], cell.y[idx, -1], cell.z[idx, -1]])
            len_seg = np.linalg.norm(end - init)
            if len_seg > 0:
                dir_seg = (end - init) / len_seg
                n_points = 10
                neur_poly3d = get_polygons_for_cylinder(init,
                                                        direction=dir_seg,
                                                        length=len_seg,
                                                        radius=cell.d[idx] / 2,
                                                        n_points=n_points,
                                                        facecolor=color,
                                                        alpha=alpha)
                for crt_poly3d in neur_poly3d:
                    ax.add_collection3d(crt_poly3d)
