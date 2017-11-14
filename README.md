# PyPoint

Fast Probabilistic Programming for Point Processes using Edward and Tensorflow

# Motivation

In this project, we explore efficient and flexible Bayesian inference for point process models using probabilistic programming. Our original interest was in modeling a publicly available dataset from the Federal Election Commission, which contains information on donations from individuals to candidates for federal office. We wanted to model the spatiotemporal dynamics of donations to answer questions like these: given the past few months of donation data, where should candidate $X$ expect to get more donations than candidate $Y$ over the next month? Where should she focus her resources?

# Models and methods we're interested in

We're interested in using GP priors for learning the distributions over intensity functions of point process models. The basic model is as follows (there are a few variations on it): 

We have points $x$ which lie in some d-dimensional space (e.g. 1D time, 2D space, or 3D space-time). Each of these points denotes a realization from a point process with some unknown rate function $f(x)$. We assume the function $f(x)$ is a random function drawn from a Gaussian Process. In the simple case where we discretize space/time and we assume a Poisson rate, we have

$f \sim GP(mu(x), K(x, x))$

$y(x) \sim Poisson(f(x))$

We're also interested in continuous time/space point processes. Inference for this is more complicated, since it requires us to infer rate values where we have not seen any events. We follow Adams et al (2009) for our inference procedure on these processes.

# Tools we're using

We use Edward's **KLqp** inference to infer a discretized Cox process model on both simulated data and our FEC dataset (see the "Simple Model" notebooks). We're also working on making this more efficient using inducing points methods.

Since black-box inference for Gaussian Processes requires computation and inversion of an $N \times N$ covariance matrix, it typically does not scale well to large datasets. To work around this issue, we implement efficient structure exploiting inference for GPs using Kronecker Methods (e.g. Flaxman et al 2015) in Tensorflow. These methods accelerate GP inference by placing constraints on the covariance matrix to make the computations involving K times faster. 

We also work on implementing efficient modifications of the algorithm from Adams et al (2009).

# What's in the files

- Simple Model Experiment(Edward).ipynb: KLqp for discretized Cox process on simulated data
- Simple Model FEC (Edward).ipynb: KLqp for discretized Cox process on FEC data
- kronecker.py: primary file for implementation of Kronecker methods
- optimizers.py, simulator.py, kronecker\_utils.py, likelihoods.py: helpers for the kronecker methods
- Kronecker Example.ipynb: example of using Kronecker inference on simulated data.
- ThinnedEvents.ipynb : example of inference of data simulated from an inhomogenuous Poisson process
- ThinnedEvents_tf.ipynb : Tensorflow implementation of the above file

# What's next

We plan to integrate the kronecker methods for use on inferring continuous time/space point process intensities. We are also working on kernel learning via marginal likelihood optimization over kernel hyperparameters, as well as implementing inducing point methods in Edward.

# References

[Adams et al, Tractable Nonparametric Bayesian Inference in Poisson Processes with Gaussian Process Intensities, Proceedings of the $$26^{th}$$ International Conference on Machine Learning, Montreal, Canada, 2009](https://hips.seas.harvard.edu/files/adams-sgcp-icml-2009.pdf)

[Flaxman et al, Fast Kronecker Inference in Gaussian Processes with non-Gaussian Likelihoods, Proceedings of the $$32^{nd}$$ International Conference on Machine Learning, Lille, France, 2015](https://www.cs.cmu.edu/~neill/papers/icml15.pdf)