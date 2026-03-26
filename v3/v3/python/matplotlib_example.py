try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib or numpy not installed, skipping")
    exit(0)

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Sine wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.savefig('plot.png')
print("Plot saved as plot.png")
import os
os.remove('plot.png')
print("Removed plot.png")
