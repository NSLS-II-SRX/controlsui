from __future__ import (unicode_literals, division, print_function,
                        absolute_import)
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
from matplotlib.figure import Figure
import numpy as np


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
