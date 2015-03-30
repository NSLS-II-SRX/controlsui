import enaml
from enaml.qt.qt_application import QtApplication
from atom.api import (Atom, observe, Typed, Int, List, Float, Bool, Dict,
                      Coerced, Enum, Unicode)
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
    points_per_update = Int(1000)
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
    roi_names = List()

    image_roi = Unicode()
    red_roi = Unicode()
    green_roi = Unicode()
    blue_roi = Unicode()

    image_roi_index = Int(0)
    red_roi_index = Int(0)
    green_roi_index = Int(0)
    blue_roi_index = Int(0)

    cs_aspect = Enum('equal', 'auto')

    _datapoints = List()
    _current_step = Int(0)

    _dirty = Bool(False)

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
        self.roi_names = sorted(self.rois.keys())
        self.image_roi = self.roi_names[0]
        self.red_roi = self.roi_names[0]
        self.green_roi = self.roi_names[0]
        self.blue_roi = self.roi_names[0]
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
        if changed['type'] == 'create':
            return
        print('rgb energy updated', changed)
        self._dirty = True

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

    def redraw(self):
        if self._dirty:
            self._dirty = False
            self._recompute_image()
            self._recompute_overlay()

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

        self._dirty = True


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

    @observe('image_roi')
    def set_roi(self, changed):
        """Helper function to set emin and emax based on a pre-determined ROI

        Parameters
        ----------
        new_roi : str
            new_roi should be a key of the `rois` dict
        """
        try:
            energies = self.rois[self.image_roi]
        except KeyError:
            return

        # suppress the changes
        self.emin = energies[0]
        self.emax = energies[1]

    _CHANNEL_MAP = {'red': 0, 'green': 1, 'blue': 2}
    _ENERGY_MAP = {'red': {'min': 'emin_red', 'max': 'emax_red'},
                   'blue': {'min': 'emin_blue', 'max': 'emax_blue'},
                   'green': {'min': 'emin_green', 'max': 'emax_green'},}

    @observe('red_roi', 'green_roi', 'blue_roi')
    def set_channel_roi(self, changed):
        channel = changed['name'].split('_')[0]
        roi = changed['value']
        try:
            energies = self.rois[roi]
        except KeyError:
            return
        emin = energies[0]
        emax = energies[1]

        # suppress the notifications so as to trigger the expensive
        # recomputing once
        setattr(self, 'emin_{}'.format(channel), emin)
        setattr(self, 'emax_{}'.format(channel), emax)

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

    def add_roi(self, roi_name, roi_tuple):
        print roi_name, roi_tuple
        # save the currently selected roi names
        red_roi = self.red_roi
        green_roi = self.green_roi
        blue_roi = self.blue_roi
        image_roi = self.image_roi
        emin_red = self.emin_red
        emax_red = self.emax_red
        emin_green = self.emin_green
        emax_green = self.emax_green
        emin_blue = self.emin_blue
        emax_blue = self.emax_blue
        # update the roi stuff
        self.roi_names.insert(0, roi_name)
        roi_names = self.roi_names
        self.rois[roi_name] = roi_tuple
        with self.suppress_notifications():
            self.roi_names = []
        self.roi_names = roi_names
        # reset the comboboxes back to the way they were
        self.image_roi_index = self.roi_names.index(image_roi)
        self.red_roi_index = self.roi_names.index(red_roi)
        self.green_roi_index = self.roi_names.index(green_roi)
        self.blue_roi_index = self.roi_names.index(blue_roi)
        # reset the energy ranges back to the way they were
        self.emin_red = emin_red
        self.emax_red = emax_red
        self.emin_blue = emin_blue
        self.emax_blue = emax_blue
        self.emin_green = emin_green
        self.emax_green = emax_green

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
    main_view.redraw_timer.start()
    app.start()


