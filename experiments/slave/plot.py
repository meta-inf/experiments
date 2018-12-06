import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Canvas:

  def __init__(self, nrow=1, ncol=1, **kw):
    kw['facecolor'] = 'w'
    self.fig = plt.figure(**kw)
    self.ax = self.fig.subplots(nrow, ncol)

  def __enter__(self):
    return self

  def __exit__(self, *args):
    plt.close(self.fig)

  def __getitem__(self, idc):
    return self.ax[idc]

  def __getattr__(self, attr):
    assert isinstance(self.ax, plt.Axes), "Multiple subplots exist"
    return getattr(self.ax, attr)

  def dump(self):
    self.fig.canvas.draw()
    data = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    return data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
