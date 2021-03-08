import neuroplotlib as nplt
import LFPy
from pathlib import Path

NPLT_PATH = Path(nplt.__path__[0])

hay_morphology = str(NPLT_PATH.parent / 'morphologies' / 'hay2011.hoc')
hall_morphology = str(NPLT_PATH.parent / 'morphologies' / 'hallerman2012.hoc')


def test_plot_neuron():
    hay_cell = LFPy.Cell(morphology=hay_morphology, pt3d=True, delete_sections=True)

    # basic plotting
    ax_xy = nplt.plot_neuron(cell=hay_cell, plane='xy')
    ax_yz = nplt.plot_neuron(cell=hay_cell, plane='yz')
    ax_xz = nplt.plot_neuron(cell=hay_cell, plane='xz')
    ax_3d = nplt.plot_neuron(cell=hay_cell, plane='3d')
    fig_proj, axes = nplt.plot_neuron(cell=hay_cell, projections3d=True, alpha=0.1)

    # with morphology
    ax_xy = nplt.plot_neuron(morphology=hay_morphology, plane='xy')

    # with args
    ax = nplt.plot_neuron(morphology=hall_morphology, plane='xy', alpha=0.2)
    ax = nplt.plot_neuron(morphology=hall_morphology, plane='xy', exclude_sections=['axon', 'my', 'node'])
    ax = nplt.plot_neuron(morphology=hall_morphology, plane='xy', xlim=[-50, 50], ylim=[-50, 200])
    ax = nplt.plot_neuron(morphology=hall_morphology, plane='xy',
                          color_soma='C0', color_dend='C1', color_apic='C2', color='black')

    hay_cell.__del__()  # avoid potential crashes due to hanging hoc refs


def test_plot_detailed_neuron():
    ax_xyd = nplt.plot_detailed_neuron(morphology=hall_morphology, plane='xy')
    ax_xzd = nplt.plot_detailed_neuron(morphology=hall_morphology, plane='xz')
    ax_yzd = nplt.plot_detailed_neuron(morphology=hall_morphology, plane='yz')
    ax_3dd = nplt.plot_detailed_neuron(morphology=hall_morphology, plane='3d')

    # with cell
    hall_cell = LFPy.Cell(morphology=hall_morphology, pt3d=True, delete_sections=True)
    ax_xyd = nplt.plot_detailed_neuron(cell=hall_cell, plane='xy')

    # with args
    ax = nplt.plot_detailed_neuron(morphology=hall_morphology, plane='xy', alpha=0.2)
    ax = nplt.plot_detailed_neuron(morphology=hall_morphology, plane='xy', exclude_sections=['axon', 'my', 'node'])
    ax = nplt.plot_detailed_neuron(morphology=hall_morphology, plane='xy', xlim=[-50, 50], ylim=[-50, 200])
    ax = nplt.plot_detailed_neuron(morphology=hall_morphology, plane='xy',
                                   color_soma='C0', color_dend='C1', color_apic='C2', color='black')

    hall_cell.__del__()  # avoid potential crashes due to hanging hoc refs


def test_plot_3d_cylinder():
    ax_3d = nplt.plot_cylinder_3d(bottom=[0, 0, 0], direction=[0, 1 ,1], length=100, radius=3, color='c',)
