import os
import h5py
import numpy as np
import itertools
import uuid
import time as ttime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)


def get_root_data():
    fname = os.path.join(os.path.expanduser('~'), 'Downloads', 'root.h5')
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

root_data = get_root_data()
# set the data from which live data will be sourced
counts = np.asarray(root_data['xrfmap/detsum/counts'])
# energy is in kev, multiply by 1000
energy = np.asarray(root_data['xrfmap/detsum/energy']) * 1000
# init the rois dict based on the hdf dataset of a root
rois = {k: v for k, v
        in zip(root_data['xrfmap/detsum/roi_name'],
               np.asarray(root_data['xrfmap/detsum/roi_limits']))}
roi_names = sorted(rois.keys())


class Plotter(object):
    def __init__(self):
        self.imfig = plt.figure()

        self.linefig = plt.figure()
        self._ax_cursor = self.linefig.add_subplot(2, 1, 1)
        self._ax_last = self.linefig.add_subplot(2, 1, 2,
                                                 sharex=self._ax_cursor)
        self._ax_cursor.set_title('Spectrum for cursor position')
        self._ax_last.set_title('Spectrum for last datapoint received')

    def new_scan(self, nx, ny, extent, channels, rois, energy):
        self._ax_cursor.cla()
        self._ax_last.cla()
        self._cursor_artists = {}
        self._last_artists = {}
        # create new plots for each
        self.roi_names = rois
        self.channel_names = channels

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
        self.channel_data = defaultdict(lambda: np.zeros((ny, nx, 4096)))
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
            self.images[data_name] = ax.imshow(np.zeros((nx, ny)),
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
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            col = int(x + 0.5)
            row = int(y + 0.5)
            # grab the data name
            # data_name = event.axes.gid
            for channel in self.channel_names:
                print('updating data for channel = %s' % channel)
                data = self.channel_data[channel][row, col, :]
                self._cursor_artists[channel].set_data(self.energy, data)

            self._ax_cursor.relim(visible_only=True)
            self._ax_cursor.autoscale_view(tight=True)
            self._ax_cursor.set_title('Cursor position: ({}, {})'.format(row,
                                                                         col))
            self._ax_cursor.legend(loc=0)
        self._draw()

    def _draw_last_spectrum(self, row, col):
        # grab the data name
        # data_name = event.axes.gid
        for channel in self.channel_names:
            print('updating data for channel = %s' % channel)
            data = self.channel_data[channel][row, col, :]
            self._last_artists[channel].set_data(self.energy, data)

        self._ax_last.relim(visible_only=True)
        self._ax_last.autoscale_view(tight=True)
        self._ax_last.set_title('Data point: ({}, {})'.format(row, col))
        self._ax_last.legend(loc=0)
        self._draw()

    def _draw(self, interval=.01):
        for fig in [self.imfig, self.linefig]:
            canvas = fig.canvas
            canvas.draw()
            plt.show(block=False)
            canvas.start_event_loop(interval)

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
                    arr[x, y, :] = v
                    # grab the summed mca spectrum
                    arr = self.images[k].get_array()
                    arr[x, y] = np.sum(v)
                    # update the image
                    self.images[k].set_array(arr)
                # add the roi data to the imshow
                elif k in rois:
                    arr = self.images[k].get_array()
                    arr[x, y] = v
                    self.images[k].set_array(arr)
            self._draw_last_spectrum(x, y)



def example_scan_that_will_drive_the_plotter(plotter):
                                             # fast_pos, start1, steps1, size1,
                                             # slow_pos, start2, steps2, size2):

    # example scan set up. this (obviously?) will not be used in production
    fast_pos = str(root_data['xrfmap/config/scan/pos1'].value)
    start1 = float(root_data['xrfmap/config/scan/start1'].value)
    step1 = float(root_data['xrfmap/config/scan/step1'].value)
    stop1 = float(root_data['xrfmap/config/scan/stop1'].value)
    slow_pos = str(root_data['xrfmap/config/scan/pos2'].value)
    start2 = float(root_data['xrfmap/config/scan/start2'].value)
    step2 = float(root_data['xrfmap/config/scan/step2'].value)
    stop2 = float(root_data['xrfmap/config/scan/stop2'].value)

    # probably want to uncomment these once this is a live scan
    # stop1 = steps1 / size1
    # stop2 = steps2 / size2

    # generate the slow/fast coordinates so that the image shape can be
    # determined
    slow_positions = np.arange(start2, stop2, step2)
    fast_positions = np.arange(start1, stop1, step1)
    image_shape = (len(slow_positions), len(fast_positions), 4096)

    # determine the extent of the image. The "extent" allows matplotlib to
    # show the pixel coordinates as the actual motor positions, instead of
    # the pixel indices
    left, right = sorted((start1,stop1))
    bottom, top = sorted((start2, stop2))

    print('start1 = %s' % start1)
    print('stop1 = %s' % stop1)
    print('step2 = %s' % step2)
    print('start2 = %s' % start2)
    print('stop2 = %s' % stop2)
    print('step2 = %s' % step2)
    print('left = %s' % left)
    print('right = %s' % right)
    print('bottom = %s' % bottom)
    print('top = %s' % top)

    # instruct the plotter to reset itself with a new image shape and extent
    plotter.new_scan(len(fast_positions), len(slow_positions),
                     extent=(left,right, bottom, top),
                     channels=['det1', 'det2', 'det3'], rois=[],
                     energy=energy)

    # do the step scan (or in this case, fake it up)
    for y_idx, slow_pos in enumerate(slow_positions):
        for x_idx, fast_pos in enumerate(fast_positions):
            print("setting data for pixel (%s, %s)" % (y_idx, x_idx))
            detvals = read_detector(y_idx, x_idx)
            # add the x and y positions
            detvals['x'] = fast_pos
            detvals['y'] = slow_pos
            # add the array indices for the x and y positions
            detvals['xidx'] = x_idx
            detvals['yidx'] = y_idx
            # mock up (most of) an event
            event = {
                'data': detvals,
                'seq_num': x_idx + y_idx * len(fast_positions),
                'uid': uuid.uuid4(),
                'time': ttime.time()
            }
            # send it as a list to support future, faster scans
            plotter.new_scan_data([event])

def set_the_data(plotter):
    plotter.new_scan(im_shape=(det1.shape[0], det1.shape[0], 3))
    plotter._rgb.red = np.sum(det1, axis=2)
    plotter._rgb.draw

print('loading det1')
det1 = np.array(root_data['xrfmap/det1/counts'])
# det1 = (det1 - det1.min()) / (det1.max() - det1.min()) * 255
print('loading det2')
det2 = np.array(root_data['xrfmap/det2/counts'])
# det2 = (det2 - det2.min()) / (det2.max() - det2.min()) * 255
print('loading det3')
det3 = np.array(root_data['xrfmap/det3/counts'])
# det3 = (det3 - det3.min()) / (det3.max() - det3.min()) * 255
# det4 = np.array(root_data['xrfmap/det4/counts'])
energy = np.array(root_data['xrfmap/detsum/energy'])

def read_detector(y, x):
    detvals = {
        'det1': det1[y, x],
        'det2': det2[y, x],
        'det3': det3[y, x],
        # 'det4': det4[y, x],
        'energy': energy
    }
    return detvals


if __name__ == "__main__":
    plotter = Plotter()
    # plt.show()
    # plotter.imfig.canvas.show()
    # plotter.linefig.canvas.show()
    # plotter.imfig.canvas.draw()
    # plotter.linefig.canvas.draw()
    # plt.pause(.1)
    example_scan_that_will_drive_the_plotter(plotter)
