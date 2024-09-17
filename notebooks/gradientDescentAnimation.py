# animation of the gradient descent technique
# the surface is the difference bewteen two multivariate gaussian functions

import matplotlib
import numpy as np
import matplotlib.cm as cm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.animation as anim

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

# gradient of a bivariate gaussian N(mu,simga)
def gradient(x,mu,sigma):
  mvx = multivariate_normal(mu, sigma).pdf(x) # probability density funciton 
  dist = (np.dot(np.linalg.inv(sigma),(x - mu))) # mahalanobis distance: measure the distance of x from the mean mu using the inverse of the covarinace matrix 
  g = mvx * dist # final gradient of the bivariate gaussian 
  return g

# step 1 - prepare the 3d plot 
x, y = np.mgrid[-3:3:.025, -2:3:.025]
grid = np.empty(x.shape + (2,))
grid[:, :, 0] = x; grid[:, :, 1] = y

# step 2 - prepare the distributions 
mu1 = np.array([0,0])
sigma1 = np.array([[1, .5], [.5, 1]])
mu2 = np.array([1,1])
sigma2 = np.array([[1.5, 0], [0, .5]])
z1 = multivariate_normal(mu1, sigma1)
z2 = multivariate_normal(mu2, sigma2)

# the function we conside is the difference of two bivariate gaussian
def f(pos): return 10*(z2.pdf(pos)-z1.pdf(pos)) # main function 
def f1(pos): return gradient(pos,mu2,sigma2)-gradient(pos,mu1,sigma1) # derivative 

# learning rate 
step = 1

# step 3 - prepare the animation 
fig = plt.figure()
CS = plt.contour(x, y, f(grid)) # the contour 
plt.clabel(CS, inline=1, fontsize=10)
line, = plt.plot([], [], lw=2) # empty line 
plt.title('Gradient Descent')

# initial position 
xdata, ydata = [.8], [1.31]
#xdata, ydata = [.8], [1.41]

def init():
    line.set_data([], [])
    return line,

def animate(i):
  pos = np.array([xdata[-1],ydata[-1]])
  delta = step*f1(pos) # gardient descent step 
  npos = pos + delta   # new position 
  xdata.append(npos[0])
  ydata.append(npos[1])
  print(f'x position: {xdata[-1]}, y position: {ydata[-1]}')
  line.set_data(xdata, ydata)
  return line,

anim = anim.FuncAnimation(fig, animate, init_func=init, frames=60, interval=1000, blit=True, repeat=False)
plt.show()





