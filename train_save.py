import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import *

equation = sys.argv[1]
lim_min = int(sys.argv[2])
lim_max = int(sys.argv[3])
step = float(sys.argv[4])
learn_pg = float(sys.argv[5])

#calc data
data = np.arange(lim_min, lim_max + step, step)
results = []
for x in data:
    results.append(eval(equation))
results= np.array(results)

#shuffle data and divide to learning & testing
s = np.arange(data.shape[0])
np.random.shuffle(s)
dat = data[s]
res = results[s]

learn_max_index = floor(len(dat)*learn_pg)

l_data = dat[:learn_max_index]
l_results = res[:learn_max_index]
t_data = dat[learn_max_index:]
t_results = res[learn_max_index:]

plt.plot(data, results, "b-")
plt.show()
