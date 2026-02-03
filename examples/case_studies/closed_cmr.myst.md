---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: examples
  language: python
  name: python3
---

(closed_cmr)=
# Estimating population size with capture-recapture

:::{post} January, 2026
:tags: ecology, hierarchical model, marginalization
:category: intermediate, howto
:author: Philip T. Patton
:::

+++

Ecologists have been estimating population sizes with capture-recapture models for over a 100 years. The method appears to have been independently developed in 1889 by C.G. Johannes Petersen, who wanted to estimate the number of European Plaice (*Pleuronectes platessa*) at his Danish marine station, and in 1930 by Fredrick C. Lincoln, who was tasked with monitoring birds across the U.S. after the passage of the Migratory Bird Treaty Act in 1918. The very first application of capture-recapture appears to have been by Laplace, who was trying to estimate the size of France's human population in 1783. Since then, researchers have developed countless forms of the basic capture-recapture model to account for complexities that arise in sampling wildlife populations. 

The goal of a most capture-recapture analyses is to estimate the size of a population, which is arguably the baseline indicator for population health. (Although note that open forms of capture-recapture exist, where the goal is to estimate survival, recruitment, and the population trajectory.) The most basic form of capture-recapture involves capturing animals, marking them, releasing them, recapturing them on a later date, and determining the number of marked animals in the second sample. The process can be continued to include many samples (i.e., not just two). Then, we can estimate the probability of recapturing an individual, $p$, which we can use to estimate the population size, $N$. 

In this example, we'll estimate the size of a meadow vole population (*Microtus pennsylvanicus*), which was surveyed by Jim Nichols of the the US Geological Survey's Eastern Ecological Research Center {cite:p}`nichols1984use`.

![](meadow-vole.jpg)

```{code-cell} ipython3
import os

import arviz.preview as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from matplotlib import colors
from matplotlib.ticker import PercentFormatter
```

:::{include} ../extra_installs.md
:::

```{code-cell} ipython3
import pymc_extras as pmx
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'  # high resolution figures
az.style.use("arviz-darkgrid")
RANDOM_SEED = 1792
rng = np.random.default_rng(RANDOM_SEED)
```

## Capture-recapture model 

{cite:t}`nichols1984use` deployed a 10 by 10 grid of traps baited with corn, and checked the traps in $T=5$ consecutive nights to see if they contained a vole. Captured voles were marked and released. They captured $n_0=56$ voles in total. The observed dataset is a binary matrix of shape $(56, 5)$ where $1$ indicates that the individual was captured on that night. Researchers will often refer to each row as the "capture history" for that individual.

```{code-cell} ipython3
try:
    vole_data = np.loadtxt(os.path.join("..", "data", "microtus.txt"))
except FileNotFoundError:
    vole_data = np.loadtxt(pm.get_data("microtus.txt"))

# unpack the two portions of the dataset
observed_dataset = vole_data[:, :-1].astype(int)
body_mass = vole_data[:, -1].astype(int)
observed_dataset[:5]
```

We are going to model these data with parameter expanded data augmentation {cite:p}`royle2009analysis`. Data augmentation works by setting an absurd upper limit to the population size, $M$. Then, we augment the observed dataset with $M-n_0$ all-zero capture histories. Each all-zero history either represents an individual that was never captured, or an individual that does not exist in the population. We can then estimate the inclusion probability $\psi$, which represents that one of the $M$ rows in the dataset represents a true individual in the population. When the detection probability is constant across occasions, we can represent the process as a zero-inflated binomial distribution 
$$
\begin{align*}
z_i &\sim \text{Bernoulli}(\psi) \\
y_i &\sim \text{Binomial}(T, z_i p)
\end{align*}
$$
where $i$ is an index for individual $i=1,2,\dots,M$, and $z_i=1$ if the individual truly exists within the population

```{code-cell} ipython3
# augment the observed data with all zero capture histories
M = 200
n0, T = observed_dataset.shape
all_zero_histories = np.zeros((M - n0, T))
augmented_dataset = np.vstack((observed_dataset, all_zero_histories))
```

## Authors
- Authored by [Benjamin T. Vincent](https://github.com/drbenvincent) in January 2023 

+++

## References
:::{bibliography}
:filter: docname in docnames
:::

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor
```

:::{include} ../page_footer.md
:::
