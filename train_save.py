import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import *

equation = sys.argv[1]
lim_min = int(sys.argv[2])
lim_max = int(sys.argv[3])
step = float(sys.argv[4])

data = np.arange(lim_min, lim_max + step, step)
results = []
for x in data:
    results.append(eval(equation))

plt.plot(data, results, "b-")
plt.show()
