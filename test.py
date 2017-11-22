import numpy as np
import GPy
import edward as ed
from kern import RBF
import tensorflow as tf
from datasets import build_toy_dataset1, build_toy_dataset2
from matplotlib import pyplot as plt
from thinnedEvents import ThinnedEventsSampler, ThinnedEventsSolver

events, Z, N, rate, measure = build_toy_dataset1()
kern = RBF(input_dim = 1, lengthscale = 10)

solver = ThinnedEventsSolver(events, kern, measure, rate)
solver.solve(10)
