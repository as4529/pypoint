import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
from IPython.display import display
import datetime
from kronecker import KroneckerSolver
from likelihoods import PoissonLike
import data_utils as sim
import numpy as np
from kernels import RBF
from grid_utils import fill_grid
from plotly import tools
from thinnedEvents_eager import run_thinnedEventsSolver, ThinnedEventsSampler
from matplotlib import pyplot as plt

fb_dons = np.genfromtxt('data/facebook_donations.csv', delimiter = ',')
dates = np.array([datetime.datetime(2015, 1, 1) + datetime.timedelta(days=x) for x in range(0, 365*2)])
#iplot([trace_fb_events])

events = fb_dons[:,0][fb_dons[:,1]>= 1.0]
events = events.reshape(-1, 1)
sampler, s_i, val = run_thinnedEventsSolver(events=events)

print(sampler.measure)
print(np.max(s_i))
plt.scatter(s_i, val)
plt.show()
