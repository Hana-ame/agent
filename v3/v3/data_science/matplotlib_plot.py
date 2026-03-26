try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib or numpy not installed, skipping")
    exit(0)

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.plot(x, y, label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine and Cosine')
plt.legend()
plt.savefig('plot.png')
print("Plot saved as plot.png")
import os
os.remove('plot.png')
print("Removed plot.png")
