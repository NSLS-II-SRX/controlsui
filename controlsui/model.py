import enaml
from enaml.qt.qt_application import QtApplication
from atom.api import (Atom, observe, Typed, Int, List, Float, Bool, Dict,
                      Coerced, Enum)
import numpy as np
import os
import h5py
import urllib
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
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
    channel_rois = Dict()
    alpha = Float()
    # display properties
    emin = Coerced(float)
    emax = Coerced(float)
    emin_red = Coerced(float)
    emax_red = Coerced(float)
    emin_blue = Coerced(float)
    emax_blue = Coerced(float)
    emin_green = Coerced(float)
    emax_green = Coerced(float)

    cs_aspect = Enum('equal', 'auto')

    _datapoints = List()
    _current_step = Int(0)

    _fig_cs = Typed(Figure)
    _fig_line = Typed(Figure)
    _ax_cursor = Typed(Axes)
    _cursor_artist = Typed(Line2D)
    _ax_last = Typed(Axes)
    _last_datapoint_artist = Typed(Line2D)

    cs = Typed(cs2d.CSOverlay)

    _energy_ranges_changed = Bool(False)

    def __init__(self, delay=None):
        super(Model, self).__init__()
        self._fig_line = Figure()
        self._fig_cs = Figure()
        # self._fig_line.set_size_inches(4, 4)
        # set up the cursor line artist
        self._ax_cursor = self._fig_line.add_subplot(2, 1, 1)
        self._ax_cursor.set_ylabel('counts')
        self._ax_cursor.set_xlabel('Energy (eV)')
        self._cursor_artist, = self._ax_cursor.plot(
            [], [], '-', label='cursor position')
        # set up the artist to look at the last datapoint
        self._ax_last = self._fig_line.add_subplot(2, 1, 2)
        self._ax_last.set_ylabel('counts')
        self._ax_last.set_xlabel('Energy (eV)')
        self._last_datapoint_artist, = self._ax_last.plot(
            [], [], '-', label='cursor position')

        self.cs = cs2d.CSOverlay(fig=self._fig_cs)
        self.cs.add_cursor_position_cb(self.new_cursor_position)

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
        self.rgba[:, :, :-1] = np.nan

    @observe('emin', 'emax')
    def _energy_changed(self, changed):
        # self.cs.update_limit_func(cs2d.absolute_limit_factory((self.emin, self.emax)))
        print('energy changed: {}'.format(changed))
        self._recompute_image()

    @observe('cs_aspect')
    def _aspect_changed(self, changed):
        self.cs._im_ax.set_aspect(changed['value'])

    @observe('emin_red', 'emax_red', 'emin_blue', 'emax_blue', 'emin_green',
             'emax_green')
    def _rgb_energy_updated(self, changed):
        self._recompute_overlay(channels=[changed['name'].split('_')[1]])

    def reset_scan(self):
        """Helper function used entirely for testing/debugging/demoing the
        srx controls ui
        """
        prev_state = self.scan_running
        if prev_state:
            self.scan_running = False
        self.live_counts[:] = np.nan
        self.rgba[:, :, :-2] = np.nan
        print self.rgba
        self._current_step = 0
        if prev_state:
            self.scan_running = True

    def new_cursor_position(self, cc, rr):
        """Callback that needs to be registered with the cross section widget

        Parameters
        ----------
        cc : int
          column
        rr : int
            row
        """
        # print('cc, rr: {}, {}'.format(cc, rr))
        counts = self.live_counts[rr, cc, :]
        # print('counts shape: {} counts values: {}'.format(counts.shape, counts))
        self._cursor_artist.set_data(self.energy, counts)
        self._ax_cursor.relim(visible_only=True)
        self._ax_cursor.autoscale_view(tight=True)
        self._ax_cursor.set_title('Cursor position: ({}, {})'.format(cc, rr))
        self._fig_line.canvas.draw()

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

        # update the last datapoint artist
        self._last_datapoint_artist.set_data(self.energy, self.live_counts[y, x, :])
        self._ax_last.set_title('Last datapoint: ({}, {})'.format(x, y))
        self._ax_last.relim(visible_only=True)
        self._ax_last.autoscale_view(tight=True)
        self._fig_line.canvas.draw()

        self._recompute_image()
        # update the overlay for all channels
        self._recompute_overlay()

    def _recompute_image(self):
        energy_mask = (self.energy > self.emin) & (self.energy < self.emax)
        data = self.live_counts[:, :, energy_mask]
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

    _CHANNEL_MAP = {'red': 0, 'green': 1, 'blue': 2}
    _ENERGY_MAP = {'red': {'min': 'emin_red', 'max': 'emax_red'},
                   'blue': {'min': 'emin_blue', 'max': 'emax_blue'},
                   'green': {'min': 'emin_green', 'max': 'emax_green'},}

    def set_channel_roi(self, new_roi, channel):
        energies = self.rois[new_roi]
        emin = energies[0]
        emax = energies[1]

        # suppress the notifications so as to trigger the expensive
        # recomputing once
        with self.suppress_notifications():
            setattr(self, self._ENERGY_MAP[channel]['min'], emin)
            setattr(self, self._ENERGY_MAP[channel]['max'], emax)

        self._recompute_overlay(channels=[channel])

    def _recompute_overlay(self, channels=None):
        """

        Helper function to recompute the overlay for one or more of the
        overlay channels

        Parameters
        ----------
        channel : {'red', 'green', 'blue'}, optional
            list of one or many of the above options
        """
        # recompute the overlay for all channels
        if channels is None:
            channels = list(self._CHANNEL_MAP.keys())

        for channel in channels:
            emin = getattr(self, self._ENERGY_MAP[channel]['min'])
            emax = getattr(self, self._ENERGY_MAP[channel]['max'])
            energy_indices = np.where(np.logical_and(self.energy > emin,
                                                     self.energy < emax))[0]
            color_data = self.live_counts[:, :, energy_indices]
            plottable = np.sum(color_data, axis=2)
            min = np.nanmin(plottable)
            max = np.nanmax(plottable)
            # print("min, max: {}, {}".format(min, max))
            if max != min:
                plottable = (plottable - min) / (max - min)
            # assign the plottable data to the correct channel
            self.rgba[:, :, self._CHANNEL_MAP[channel]] = plottable
        # trigger the cross section mpl object to update the overlay
        self.cs.update_overlay(self.rgba)


    @observe('alpha')
    def _alpha_changed(self, changed):
        self.rgba[:, :, 3] = self.alpha
        self.cs.update_overlay(self.rgba)
        print self.rgba


if __name__ == "__main__":
    with enaml.imports():
        from controlsui.mockup import Main

    app = QtApplication()
    main_view = Main()
    main_view.show()
    app.start()


