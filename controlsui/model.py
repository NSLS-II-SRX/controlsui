from atom.api import Atom, observe, Typed, Int, List, Float, Bool
import numpy as np
import os
import h5py
import urllib
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
import itertools
import logging


logger = logging.getLogger()


def get_root_data():
    fname = '/home/edill/Downloads/coot.h5'
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
    delay = Int(10)
    points_per_update = Int(100)
    scan_running = Bool(False)

    # display properties
    emin = Float(0)
    emax = Float(1)
    _datapoints = List()
    _current_step = Int(0)

    _fig = Typed(Figure)
    _ax = Typed(Axes)
    _axes_image = Typed(AxesImage)


    def __init__(self, delay=None):
        super(Model, self).__init__()
        self._fig = Figure()
        self._ax = self._fig.add_subplot(111, aspect='equal')
        self._ax.set_xlabel('x position')
        self._ax.set_ylabel('y position')

        root_data = get_root_data()
        self.counts = np.asarray(root_data['xrfmap/detsum/counts'])
        self.energy = np.asarray(root_data['xrfmap/detsum/energy'])
        if delay is not None:
            self.delay = delay

    @observe('counts')
    def _counts_changed(self, changed):
        xshape = self.counts.shape[0]
        yshape = self.counts.shape[1]
        self._datapoints = [(x, y) for x, y
                            in itertools.product(range(xshape), range(yshape))]
        self._ax.set_xlim(0, xshape)
        self._ax.set_ylim(0, yshape)
        self.live_counts = np.zeros(self.counts.shape)
        # print('state: {}'.format(self.__getstate__()))

    def reset_scan(self):
        self.scan_running = False
        self.live_counts[:] = np.nan
        self._current_step = 0
        self.scan_running = True

    def new_data(self):
        for i in range(self.points_per_update):
            if self._current_step >= len(self._datapoints):
                self.scan_running = False
                break
            (x, y) = self._datapoints[self._current_step]
            print('adding data for (x, y): ({}, {})'.format(x, y))
            self.live_counts[x, y, :] = self.counts[x, y, :]
            self._current_step += 1
        energy_indices = np.where(np.logical_and(self.energy > self.emin,
                                                 self.energy < self.emax))[0]
        data = self.live_counts[:, :, energy_indices]
        plottable = np.sum(data, axis=2).T
        try:
            self._axes_image.set_data(plottable)
        except AttributeError:
            self._axes_image = self._ax.imshow(plottable)

        self._ax.set_aspect('equal')
        self._ax.figure.canvas.draw()

