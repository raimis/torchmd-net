'''
Interactive visualizer for PES, trajectories and structure
needs:
conda install nglview -c conda-forge
conda install -c conda-forge ipympl -y

and set "%matplotlib widget" at the begining of the notebook


A tutorial on using widget:

https://kapernikov.com/ipywidgets-with-matplotlib/
'''
import matplotlib.pyplot as plt
from ipywidgets import HBox, VBox
import ipywidgets as widgets
import nglview
import numpy as np
from matplotlib.collections import LineCollection
import mdtraj
import pyemma

def set_viewer_options(iwdg):
    iwdg.add_spacefill()
    iwdg.add_cartoon(color_scheme='residueindex')
    iwdg.update_spacefill(radiusType='covalent',
                                       scale=0.6,
                                       color_scheme='resname')
    iwdg.center()
    return iwdg

def setup_viewer(sim_traj):
    iwdg = nglview.NGLWidget()
    iwdg.add_trajectory(sim_traj)
    iwdg._remote_call('setSize', target='Widget',
                                   args=['%dpx' % (400,), '%dpx' % (250,)])
    iwdg.player.delay = 200.0
    return set_viewer_options(iwdg)

def link_ngl_wdgt_to_ax_pos_(ax,lineh,linev,dot, pos, ngl_widget):
    from matplotlib.widgets import AxesWidget
    from scipy.spatial import cKDTree
    r"""
    Initial idea for this function comes from @arose, the rest is @gph82 and @clonker
    """

    kdtree = cKDTree(pos)
    x, y = pos.T

    ngl_widget.isClick = False

    def onclick(event):
        linev.set_xdata((event.xdata, event.xdata))
        lineh.set_ydata((event.ydata, event.ydata))
        data = [event.xdata, event.ydata]
        _, index = kdtree.query(x=data, k=1)
        dot.set_xdata((x[index]))
        dot.set_ydata((y[index]))
        ngl_widget.isClick = True
        ngl_widget.frame = index

    def my_observer(change):
        r"""Here comes the code that you want to execute
        """
        ngl_widget.isClick = False
        _idx = change["new"]
        try:
            dot.set_xdata((x[_idx]))
            dot.set_ydata((y[_idx]))
        except IndexError as e:
            dot.set_xdata((x[0]))
            dot.set_ydata((y[0]))
            print("caught index error with index %s (new=%s, old=%s)" % (_idx, change["new"], change["old"]))

    # Connect axes to widget
    axes_widget = AxesWidget(ax)
    axes_widget.connect_event('button_release_event', onclick)

    # Connect widget to axes
    ngl_widget.observe(my_observer, "frame", "change")


def get_segments(points):
    n_timesteps, dim = points.shape
    segments = np.concatenate([points[:-1].reshape(-1, 1, dim), points[1:].reshape(-1, 1, dim)], axis=1)
    return segments

def plot_tica_trajectory(points, ax):
    segments = get_segments(points)
    n_timesteps, dim = points.shape
    color = np.arange(n_timesteps)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(0, n_timesteps)

    lc = LineCollection(segments, cmap='cividis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(color)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    st = ax.plot([points[0,0]], [points[0,1]], marker='*',markerfacecolor='b', markersize=15, markeredgecolor='w')
    nd = ax.plot([ points[-1,0]], [points[-1,1]], marker='p',markerfacecolor='r', markersize=15, markeredgecolor='w')
    return line, st, nd


def plot_trajectories_on_pes(traj, dimred_features, topology, start_index=0):
    '''Interactive plot of a collection of trajectories on the underlying PES with structure visualization.

    The star is the starting point of the trajectory and the pentagone is its end.
    
    Args:
        traj (np.array): collection of trajectories. dim should be (n_traj, n_steps, n_atoms, 3)
        dimred_features (np.array): features associated with the trajectories. dim should be (n_traj, n_steps, 2)
        topology (mdtraj.Topology): Topology of the structure

    '''

    idx = start_index
    n_traj = traj.shape[0]
    # /10 to convert positions from ang to nano
    sim_traj = mdtraj.Trajectory(traj[idx]/10, topology)
    tica_concatenated = np.concatenate(dimred_features)
    output = widgets.Output()
    with output:
        fig, ax = plt.subplots(constrained_layout=True, figsize=(5,3))
    # move the toolbar to the bottom
    fig.canvas.toolbar_position = 'bottom'
    # plot PES associated with the dimentionality reduced features
    _, _, _ = pyemma.plots.plot_free_energy(*tica_concatenated[:, :2].T,ax=ax, legacy=False)
    ax.set_xlabel('IC 1')
    ax.set_ylabel('IC 2')

    iwdg = setup_viewer(sim_traj)
    # setup the location ofthe visualized frame
    lineh = ax.axhline(ax.get_ybound()[0], c="black", ls='--')
    linev = ax.axvline(ax.get_xbound()[0], c="black", ls='--')
    dot, = ax.plot(dimred_features[idx][0,0],dimred_features[idx][0,1], 'o', c='red', ms=7)

    link_ngl_wdgt_to_ax_pos_(ax,lineh,linev,dot, dimred_features[idx], iwdg)

    line, st, nd = plot_tica_trajectory(dimred_features[idx], ax)

    def update_trajectory(change):
        idx = change.new
        # update trajectory trace
        line.set_segments(get_segments(dimred_features[idx]))
        st[0].set_xdata(dimred_features[idx][0,0])
        st[0].set_ydata(dimred_features[idx][0,1])
        nd[0].set_xdata(dimred_features[idx][-1,0])
        nd[0].set_ydata(dimred_features[idx][-1,1])
        # update molecule display
        sim_traj = mdtraj.Trajectory(traj[idx]/10, topology)
        idd = iwdg._ngl_component_ids[0]
        iwdg.remove_component(idd)
        iwdg.add_trajectory(sim_traj)
        # update the visualization settings
        iwdg.clear()
        set_viewer_options(iwdg)
        # update link between molecule display and trajectory trace
        link_ngl_wdgt_to_ax_pos_(ax,lineh,linev,dot, dimred_features[idx], iwdg)
    # slider to change trajectory
    int_slider = widgets.BoundedIntText(value=0, min=0, max=n_traj-1, step=1, description='Trajectory index: ')
    int_slider.observe(update_trajectory, 'value')

    controls = HBox([output, VBox([int_slider,iwdg])])
    return controls
