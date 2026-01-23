import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

DX = X + Y
DY = -X + Y

plt.figure(figsize=(8,6))

plt.quiver(X, Y, DX, DY)

plt.xlabel('x')
plt.ylabel('y')
plt.title(r'2D Phase Space: $\dot{x}=x+y, \dot{y}=-x+y$')
plt.grid(True)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()
