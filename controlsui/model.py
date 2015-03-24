import enaml
from enaml.qt.qt_application import QtApplication
from atom.api import Atom, observe, Typed, Int, List, Float, Bool, Dict, Coerced
import numpy as np
import os
import h5py
import urllib
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
import itertools
import logging
from bubblegum.backend.mpl import cross_section_2d as cs2d

logger = logging.getLogger()


def get_root_data():
    fname = '/home/edill/Downloads/root.h5'
    url = 'http://cars.uchicago.edu/gsecars/data/Xspress3Data/Root.h5'
    if os.path.exists(fname):
        mapfile = h5py.File(fname)
        logger.debug('loaded data from disk')
    else:
        data = urllib.urlretrieve(url, fname)
        fname = data[0]
        logger.debug('downloaded data from {}'.format(url))
        mapfile = h5py.File(fname)

    return mapfile


class Model(Atom):
    counts = Typed(np.ndarray)
    energy = Typed(np.ndarray)
    live_counts = Typed(np.ndarray)
    rgba = Typed(np.ndarray)
    delay = Int(10)
    points_per_update = Int(100)
    scan_running = Bool(False)
    rois = Dict()
    cs = Typed(cs2d.CSOverlay)
    channel_rois = Dict()
    alpha = Float()
    # display properties
    emin = Coerced(float)
    emax = Coerced(float)
    _datapoints = List()
    _current_step = Int(0)

    _fig = Typed(Figure)

    _energy_ranges_changed = Bool(False)

    def __init__(self, delay=None):
        super(Model, self).__init__()
        self._fig = Figure()
        self.cs = cs2d.CSOverlay(fig=self._fig)

        root_data = get_root_data()
        self.counts = np.asarray(root_data['xrfmap/detsum/counts'])
        # energy is in kev, multiply by 1000
        self.energy = np.asarray(root_data['xrfmap/detsum/energy']) * 1000
        self.rois = {k: v for k, v
                     in zip(root_data['xrfmap/detsum/roi_name'],
                            np.asarray(root_data['xrfmap/detsum/roi_limits']))}
        print('rois: {}'.format(self.rois))
        if delay is not None:
            self.delay = delay

        try:
            self.emin = 0
        except AttributeError:
            # thrown because the canvas is not yet created
            pass
        try:
            self.emax = 100
        except AttributeError:
            # thrown because the canvas is not yet created
            pass

        self.alpha = 0.5

    @observe('counts')
    def _counts_changed(self, changed):
        xshape = self.counts.shape[1]
        yshape = self.counts.shape[0]
        self._datapoints = [(x, y) for x, y
                            in itertools.product(range(xshape), range(yshape))]
        self.live_counts = np.zeros(self.counts.shape)
        self.live_counts[:] = np.nan
        self.rgba = np.zeros(self.counts.shape[:2] + (4,))
        self.rgba[:] = 0

    @observe('emin', 'emax')
    def _energy_changed(self, changed):
        # self.cs.update_limit_func(cs2d.absolute_limit_factory((self.emin, self.emax)))
        print('energy changed: {}'.format(changed))
        self._recompute_image()

    def reset_scan(self):
        """Helper function used entirely for testing/debugging/demoing the
        srx controls ui
        """
        prev_state = self.scan_running
        if prev_state:
            self.scan_running = False
        self.live_counts[:] = np.nan
        self.rgba[:] = 0
        self._current_step = 0
        if prev_state:
            self.scan_running = True

    def new_data(self):
        new_datapoints = []
        for i in range(self.points_per_update):
            if self._current_step >= len(self._datapoints):
                self.scan_running = False
                break
            (x, y) = self._datapoints[self._current_step]
            new_datapoints.append((x,y))
            # print('adding data for (x, y): ({}, {})'.format(x, y))
            self.live_counts[y, x, :] = self.counts[y, x, :]
            self._current_step += 1

        self._recompute_image()

    def _recompute_image(self):
        energy_indices = np.where(np.logical_and(self.energy > self.emin,
                                                 self.energy < self.emax))[0]
        data = self.live_counts[:, :, energy_indices]
        plottable = np.sum(data, axis=2)

        self.cs.update_image(plottable)
        try:
            self.cs._update_artists()
        except TypeError as te:
            # i think this is only thrown at the beginning
            print te

    def set_roi(self, new_roi):
        """Helper function to set emin and emax based on a pre-determined ROI

        Parameters
        ----------
        new_roi : str
            new_roi should be a key of the `rois` dict
        """
        energies = self.rois[new_roi]

        # suppress the changes
        self.emin = energies[0]
        self.emax = energies[1]

    _channel_map = {'red': 0, 'green': 1, 'blue': 2}

    def set_channel_roi(self, new_roi, channel):
        energies = self.rois[new_roi]
        emin = energies[0]
        emax = energies[1]

        energy_indices = np.where(np.logical_and(self.energy > emin,
                                                 self.energy < emax))[0]
        color_data = self.live_counts[:, :, energy_indices]
        plottable = np.sum(color_data, axis=2)
        plottable -= np.nanmin(plottable)
        plottable /= np.nanmax(plottable)

        self.rgba[:, :, self._channel_map[channel]] = plottable
        self.cs.update_overlay(self.rgba)

    @observe('alpha')
    def _alpha_changed(self, changed):
        if changed['type'] == 'create':
            return
        self.rgba[:, :, 3] = self.alpha
        self.cs.update_overlay(self.rgba)
        print self.rgba

    def _compute_rgba_overlay(self, channel):
        energy_range = self.channel_rois.get(channel, None)




if __name__ == "__main__":
    with enaml.imports():
        from controlsui.mockup import Main

    app = QtApplication()
    main_view = Main()
    main_view.show()
    app.start()


