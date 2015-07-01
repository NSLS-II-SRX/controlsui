from __future__ import (unicode_literals, division, print_function,
                        absolute_import)

import enaml
from enaml.qt.qt_application import QtApplication
with enaml.imports():
    from controlsui.mk2.mockup import MainView, Model

from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
import numpy as np
import time as ttime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from collections import defaultdict
from matplotlib.figure import Figure

channel_map = {'red': 0, 'green': 1, 'blue': 2}

def auto_redraw(func):
    def inner(self, *args, **kwargs):
        if self.figure is None:
            return
        force_redraw = kwargs.pop('force_redraw', None)
        if force_redraw is None:
            force_redraw = self._auto_redraw

        ret = func(self, *args, **kwargs)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        return ret

    inner.__name__ = func.__name__
    inner.__doc__ = func.__doc__

    return inner


class ManagedRGBAxes(RGBAxes):
    @classmethod
    def from_one(cls, rgb, fig=None, auto_redraw=True):
        # create an instance from an MxNx3 array
        if fig is None:
            fig = Figure()
        cls = ManagedRGBAxes(fig, [0.1, 0.1, 0.8, 0.8], pad=0.0)
        r = rgb[:, :, 0]
        g = rgb[:, :,1]
        b = rgb[:, :,2]
        im = cls.imshow_rgb(r=r, g=g, b=b, origin='lower')
        # unpack the four imshow images into class attributes
        cls.imrgb, cls.imr, cls.img, cls.imb = im
        cls._auto_redraw = auto_redraw
        return cls

    @classmethod
    def from_three(cls, r, g, b, fig=None, auto_redraw=True):
        # create an instance from three independent channels
        if fig is None:
            fig = Figure()
        cls = ManagedRGBAxes(fig, [0.1, 0.1, 0.8, 0.8], pad=0.0)
        im = cls.imshow_rgb(r=r, g=g, b=b, origin='lower')
        # unpack the four imshow images into class attributes
        cls.imrgb, cls.imr, cls.img, cls.imb = im
        cls._auto_redraw = auto_redraw
        return cls

    @property
    def shape(self):
        return self.rgb.shape

    @shape.setter
    def shape(self, new_shape):
        if new_shape != self.shape:
            self._updater(rgb=np.zeros(new_shape),
                          r=np.zeros(new_shape),
                          g=np.zeros(new_shape),
                          b=np.zeros(new_shape))

    def _updater(self, rgb=None, r=None, g=None, b=None):
        """

        Parameters
        ----------
        rgb : MxNx3
        r : MxNx3
            filled in [:,:,0]
        g : MxNx3
            filled in [:,:,1]
        b : MxNx3
            filled in [:,:,2]
        """
        if r is not None:
            self.imr.set_array(r)
        if g is not None:
            self.img.set_array(g)
        if b is not None:
            self.imb.set_array(b)
        if rgb is not None:
            self.imrgb.set_array(rgb)

    def _format(self, arr, channel):
        """
        Parameters
        ----------
        arr : MxN array
        channel : {'R', 'G', 'B', 'RGB'}
        """
        data = getattr(self, channel)
        data[:, :, channel_map[channel]] = arr
        return data

    @property
    def extent(self):
        return self.imrgb.get_extent()

    @extent.setter
    def extent(self, extent):
        self.imrgb.set_extent(extent)
        self.imr.set_extent(extent)
        self.img.set_extent(extent)
        self.imb.set_extent(extent)

    @property
    def red(self):
        return self.imr.get_array().data

    @red.setter
    @auto_redraw
    def red(self, red):
        # update red and rgb
        r = self._format(red, 'red')
        rgb = self.imrgb.get_array().data
        rgb[:, :, 0] = red
        self._updater(r=r, rgb=rgb)

    @property
    def green(self):
        return self.img.get_array().data

    @green.setter
    @auto_redraw
    def green(self, green):
        # update green and rgb
        g = self._format(green, 'green')
        rgb = self.imrgb.get_array().data
        rgb[:, :, 1] = green
        self._updater(g=g, rgb=rgb)

    @property
    def blue(self):
        return self.imb.get_array().data

    @blue.setter
    @auto_redraw
    def blue(self, blue):
        # update blue and rgb
        b = self._format(blue, 'blue')
        rgb = self.imrgb.get_array().data
        rgb[:, :, 2] = blue
        self._updater(b=b, rgb=rgb)

    @property
    def rgb(self):
        return self.imrgb.get_array().data

    @rgb.setter
    @auto_redraw
    def rgb(self, rgb):
        # update red, green blue and rgb
        self.shape = rgb.shape
        r = self._format(rgb[:, :, 0], 'red')
        g = self._format(rgb[:, :, 1], 'green')
        b = self._format(rgb[:, :, 2], 'blue')
        self._updater(rgb=rgb, r=r, g=g, b=b)

    @property
    def figure(self):
        return self.RGB.figure


class Plotter(object):
    """Plotter that shows an MxN grid of raster scan images and two mca spectra

    Attributes
    ----------
    draw_interval : int
        The time (in seconds) between calls to update the GUI
    imfig : matplotlib.figure.Figure
        The figure that holds the MxN grid of images
    linefig : matplotlib.figure.Figure
        The figure that holds the 2x1 (rr x cc) grid of mca spectra
    model : atom.Atom
        The Atom model that holds the basic information that the GUI needs
    ui : enaml.widgets.MainView
        The Enaml widget that holds the MxN image figure, the 2x1 line
        figure and the image scaling controls
    _ax_cursor : matplotlib.axes.Axes
        The axes that draws the artists that show the mca spectrum for the
        location of the cursor
    _ax_last : matplotlib.axes.Axes
        The axes that draws the artists that show the mca spectrum for the
        last received datapoint
    _cursor_artists : dict
        The dictionary of line artists that draw the mca spectrum for each
        detector based on the cursor position
    _last_artists : dict
        The dictionary of line artists that draw the mca spectrum for each
        detector based on the last received datapoint
    self.nx : int
        The number of datapoints in the x direction (cols)
    self.ny : int
        The number of datapoints in the y direction (rows)
    self.roi_names : list
        List of roi names. These roi names must be the data keys in the event
        dictionary that gets passed to the `new_scan_data` function
    self.channel_names : list
        List of channel names. These are the detectors. These names must be
        the data keys in the event dictionary that gets passed to the
        `new_scan_data` function. These names are also the keys to the
        `_cursor_artists` and `last_artists` dicts.
    energy : list
        The energy of the mca channels. Must (obviously?) be the same length
        as the mca data
    images : dict
        The return value from the imshow calls in the imfig axes grid
        (type is matplotlib.images.AxesImage).
        Dict keys are self.roi_names + self.channel_names
    channel_data : defaultdict
        Dict keyed on self.channel_names. Shape is (ny, nx, len(energy))
    grid : mpl_toolkits.axes_grid1.AxesGrid
    """

    def __init__(self):
        try:
            self.app = QtApplication()
        except RuntimeError:
            # only one QApp can exist
            pass
        self.draw_interval = 1 # second
        self.last_draw_time = ttime.time()
        self.imfig = Figure()
        self.linefig = Figure()
        self.model = Model(imfig=self.imfig, linefig=self.linefig)
        self.ui = MainView(model=self.model)
        self.ui.show()

        self._ax_cursor = self.linefig.add_subplot(2, 1, 1)
        self._ax_last = self.linefig.add_subplot(2, 1, 2,
                                                 sharex=self._ax_cursor)
        self._ax_cursor.set_title('Spectrum for cursor position')
        self._ax_last.set_title('Spectrum for last datapoint received')

    def new_scan(self, nx, ny, extent, channels, rois, energy):
        self.nx = nx + 1
        self.ny = ny + 1
        self._ax_cursor.cla()
        self._ax_last.cla()
        self._cursor_artists = {}
        self._last_artists = {}
        # create new plots for each
        self.roi_names = rois
        self.channel_names = channels
        self.model.image_names = rois + channels

        for channel in self.channel_names:
            self._cursor_artists[channel], = self._ax_cursor.plot(
                [], [], '-', label=channel)
            self._last_artists[channel], = self._ax_last.plot(
                [], [], '-', label=channel)
        # turn on the legends
        self._ax_last.legend(loc=0)
        self._ax_cursor.legend(loc=0)

        self.energy = energy
        num_im = len(self.channel_names) + len(rois)
        data_labels = self.channel_names + self.roi_names
        self.images = {}
        self.channel_data = defaultdict(
            lambda: np.zeros((self.ny, self.nx, 4096)))
        nrows = int(np.ceil(np.sqrt(num_im)))
        ncols = int(np.ceil(num_im / float(nrows)))
        # clear the iamge
        self.imfig.clf()
        self.grid = AxesGrid(
            self.imfig, 111,  # similar to subplot(144)
            nrows_ncols=(nrows, ncols),
            axes_pad=0.15,
            label_mode="1",
            share_all=True,
            # cbar_location="top",
            # cbar_mode="each",
            # cbar_size="7%",
            # cbar_pad="2%",
        )

        for data_name, ax in zip(data_labels, self.grid.axes_all):
            print('adding imshow for %s' % data_name)
            ax.annotate(data_name, (.5, 1), xycoords="axes fraction")
            ax.gid = data_name
            self.images[data_name] = ax.imshow(np.zeros((self.nx, self.ny)),
                                          extent=extent,
                                          interpolation='nearest')
        self.imfig.canvas.mpl_connect('motion_notify_event',
                                      self.new_cursor_position)

        self._draw()

    def new_cursor_position(self, event):
        """Callback that needs to be registered with the cross section widget

        Parameters
        ----------
        cc : int
          column
        rr : int
            row
        """
        # raise Exception("foo")
        x, y = event.xdata, event.ydata
        print('(x, y) = (%s, %s)' % (x, y))
        if x is not None and y is not None:
            col = int(x + 0.5)
            row = int(y + 0.5)
            print('(col, row) = (%s, %s)' % (col, row))
            # grab the data name
            # data_name = event.axes.gid
            for channel in self.channel_names:
                # print('updating data for channel = %s' % channel)
                data = self.channel_data[channel][row, col, :]
                self._cursor_artists[channel].set_data(self.energy, data)

            self._ax_cursor.relim(visible_only=True)
            self._ax_cursor.autoscale_view(tight=True)
            self._ax_cursor.set_title('Cursor position: ({}, {})'.format(row,
                                                                         col))
            self._ax_cursor.legend(loc=0)
        self._draw()

    def _draw(self, interval=.01):
        if ttime.time() - self.last_draw_time > self.draw_interval:
            print('redrawing plot')
            for fig in [self.imfig, self.linefig]:
                canvas = fig.canvas
                canvas.draw()
                plt.show(block=False)
                canvas.start_event_loop(interval)
                self.last_draw_time = ttime.time()

    def new_scan_data(self, events):
        """Callback that accepts events or event-like dicts

        Expected fields:
        data = {
            'xidx'
            'yidx'

        time (float)
        seq_num (int)
        """
        for event in events:
            datadict = event['data']
            x = datadict['xidx']
            y = datadict['yidx']
            for k, v in datadict.items():
                # stash the channel data
                if k in self.channel_names:
                    arr = self.channel_data[k]
                    # store the raw mca spectrum
                    arr[y, x, :] = v
                    # grab the summed mca spectrum
                    arr = self.images[k].get_array()
                    arr[x, y] = np.sum(v)
                    # update the image
                    self.images[k].set_array(arr)
                    # keep the scaling synchronized with the Atom model/view
                    if self.model.autoscale[self.model.image_names.index(k)]:
                        clim = (arr.min(), arr.max())
                        self.model.clim_min[
                            self.model.image_names.index(k)] = clim[0]
                        self.model.clim_max[
                            self.model.image_names.index(k)] = clim[1]
                    else:
                        i0 = self.model.clim_min[
                            self.model.image_names.index(k)]
                        i1 = self.model.clim_max[
                            self.model.image_names.index(k)]
                        clim = (i0, i1)
                    self.images[k].set_clim(clim)

                # add the roi data to the imshow
                elif k in self.roi_names:
                    arr = self.images[k].get_array()
                    arr[x, y] = v
                    # raise Exception()
                    self.images[k].set_array(arr)
                    # keep the scaling synchronized with the Atom model/view
                    if self.model.autoscale[self.model.image_names.index(k)]:
                        clim = (arr.min(), arr.max())
                        self.model.clim_min[
                            self.model.image_names.index(k)] = clim[0]
                        self.model.clim_max[
                            self.model.image_names.index(k)] = clim[1]
                    else:
                        i0 = self.model.clim_min[
                            self.model.image_names.index(k)]
                        i1 = self.model.clim_max[
                            self.model.image_names.index(k)]
                        clim = (i0, i1)
                    self.images[k].set_clim(clim)

            self._draw_last_spectrum(y, x)

    def _draw_last_spectrum(self, row, col):
        # grab the data name
        # data_name = event.axes.gid
        for channel in self.channel_names:
            # print('updating data for channel = %s' % channel)
            data = self.channel_data[channel][row, col, :]
            self._last_artists[channel].set_data(self.energy, data)

        self._ax_last.relim(visible_only=True)
        self._ax_last.autoscale_view(tight=True)
        self._ax_last.set_title('Data point: ({}, {})'.format(row, col))
        self._ax_last.legend(loc=0)
        self._draw()
