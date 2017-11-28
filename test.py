import numpy as np
import GPy
import edward as ed
from kern import RBF
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
from datasets import build_toy_dataset1, build_toy_dataset2
from matplotlib import pyplot as plt
from thinnedEvents_eager import ThinnedEventsSampler, ThinnedEventsSolver

events, Z, N, rate, measure = build_toy_dataset1()
kern = RBF(input_dim = 1, lengthscale = 10)
sampler = ThinnedEventsSampler(kern, rate=2)
x_K, y_K, x_M, y_M = sampler.run()
plt.scatter(x_K[:,0], x_K[:,1])
plt.show()
#solver = ThinnedEventsSolver(events, kern, measure, rate)
#solver.solve(10)
