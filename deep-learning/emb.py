import numpy as np
import matplotlib.pyplot as plt

sample_space = np.linspace(1, 25, 1000)
Z = np.sum(np.exp(-sample_space))

f = np.exp(-sample_space) / Z
print(np.sum(f))
print(f)
plt.plot(
    sample_space, np.exp(-sample_space) / Z
)
plt.show()