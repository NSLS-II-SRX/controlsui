from __future__ import (division, unicode_literals, print_function,
                        absolute_import)
from controlsui.common import Plotter
import os
import h5py
import numpy as np
import uuid
import time as ttime
import urllib

import logging
logger = logging.getLogger(__name__)


def get_root_data():
    fname = os.path.join(os.path.expanduser('~'), 'Downloads', 'root.h5')
    url = 'http://cars.uchicago.edu/gsecars/data/Xspress3Data/Root.h5'
    if os.path.exists(fname):
        mapfile = h5py.File(fname)
        print('loaded data from disk')
    else:
        print('downloading data from {}'.format(url))

        def printer(blocknum, blocksize, totalsize):
            percent = np.round(blocknum * blocksize / totalsize*100)
            if percent > printer.percent:
                printer.percent = percent
                done = np.round(blocknum * blocksize / 1e6)
                total = np.round(totalsize / 1e6)
                print("%s%% done. %sMB downloaded of %sMB" % (percent, done,
                                                              total))
        printer.percent = 0
        data = urllib.urlretrieve(url, fname, reporthook=printer)
        fname = data[0]
        print('downloaded data from {}'.format(url))
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
                     channels=['det1', 'det2', 'det3', 'det4'],
                     rois=['roi590:670'],
                     energy=energy)

    # do the step scan (or in this case, fake it up)
    for y_idx, slow_pos in enumerate(slow_positions):
        for x_idx, fast_pos in enumerate(fast_positions):
            print("setting data for pixel (%s, %s)" % (y_idx, x_idx))
            detvals = read_detector(y_idx, x_idx)
            # add the x and y positions
            detvals['x'] = fast_pos
            detvals['y'] = slow_pos
            detvals['roi590:670'] = np.sum(detvals['det1'][590:670]+
                                           detvals['det2'][590:670]+
                                           detvals['det3'][590:670]+
                                           detvals['det4'][590:670])
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


print('loading det1')
det1 = np.array(root_data['xrfmap/det1/counts'])
print('loading det2')
det2 = np.array(root_data['xrfmap/det2/counts'])
print('loading det3')
det3 = np.array(root_data['xrfmap/det3/counts'])
print('loading det4')
det4 = np.array(root_data['xrfmap/det4/counts'])
# det1 = (det1 - det1.min()) / (det1.max() - det1.min()) * 255
# det2 = (det2 - det2.min()) / (det2.max() - det2.min()) * 255
# det3 = (det3 - det3.min()) / (det3.max() - det3.min()) * 255
energy = np.array(root_data['xrfmap/detsum/energy'])

def read_detector(y, x):
    detvals = {
        'det1': det1[y, x],
        'det2': det2[y, x],
        'det3': det3[y, x],
        'det4': det4[y, x],
        'energy': energy
    }
    return detvals


if __name__ == "__main__":
    plotter = Plotter()
    plotter.ui.show()
    # plt.show()
    # plotter.imfig.canvas.show()
    # plotter.linefig.canvas.show()
    # plotter.imfig.canvas.draw()
    # plotter.linefig.canvas.draw()
    # plt.pause(.1)
    # example_scan_that_will_drive_the_plotter(plotter)
