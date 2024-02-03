import latfun
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10**(-5), 10**(-5), 10**7)
dos = latfun.square.dos(x)
plt.plot(x, dos)
plt.show()


