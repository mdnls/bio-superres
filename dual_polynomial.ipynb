from numpy.core.function_base import linspace
import cvxpy as cp
from cvxpy.atoms.affine.bmat import bmat
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import cmath

def make_signal(n_spikes, amplitude_noise): 
    spikes = (np.arange(n_spikes)) / n_spikes
    phases = np.random.uniform(size=(n_spikes,))
    amplitudes = np.random.normal(size=(n_spikes,))
    return spikes, amplitudes * np.exp(2 * np.pi * 1j * phases)

def plot_signal(signal):
    spikes, vals = signal
    plt.plot(spikes, np.real(vals))
    plt.plot([0, 1], [0, 0], "r--")

def measure_signal(signal, cutoff):
    spikes, vals = signal
    measure_freq = lambda k: np.dot(vals, np.exp(-2 * np.pi * 1j * k * spikes))
    return np.array([measure_freq(k) for k in np.arange(-cutoff, cutoff+1)]).reshape((-1, 1))

def evaluate_dual(X, fourier_coeffs):
    f_c = (fourier_coeffs.shape[0] - 1) // 2
    n = X.shape[0]
    idft = np.array([np.exp(-2 * np.pi * 1j * k * X[i]) for i in range(n) for j in range(-f_c, f_c+1)])

def myplot(c_opt):
    n = len(c_opt)
    f_c = n//2
    freqs = range(-f_c,f_c+1)
    def value(t):
      res = 0
      for i in range (-f_c,f_c+1):
        res += c_opt[i]*np.exp(2*np.pi*1j*t*i)
      return 1 - abs(res)
    x = linspace(0,1,1000)
    plt.plot(x,value(x))

n_spikes = 5
f_c = 3*n_spikes
d = 2*f_c + 1

SIGNAL = make_signal(n_spikes, 1)
#spikes = np.array([0.1,0.2,0.3,0.8,0.9])
#vals = np.array([1.0,-1.0,1.0,-1.0,1.0])
#SIGNAL = spikes,vals

Y = measure_signal(SIGNAL, f_c)
Q = cp.Variable((d, d), hermitian=True) # is_hermitian makes Q complex automatically
c = cp.Variable((d, 1), complex=True)


constraints = [bmat([[Q, c], [c.H, np.eye(1)]]) >> 0] + \
    [cp.trace(np.eye(d, k=j).T @ Q) == (1 if j==0 else 0) for j in range(d)]

objective = cp.Maximize(cp.real(c.H @ Y))
prob = cp.Problem(objective, constraints)
prob.solve()
print(prob.status)
print("Objective value", objective.value)
c_opt = c.value
tv_norm = 0
spikes, vals = SIGNAL
for u in vals:
  tv_norm += abs(u)

print("TV norm", tv_norm)  #we compare the TV_norm value with the value of the objective function to make sure the program is solved
u = []                    #build coefficient vector u for equation |q(x)|^2 - 1 = 0
for i in range(-2*f_c,2*f_c +1):
  res = 0
  for k in range(-f_c, f_c + 1):
    if ((k-i)>= -f_c and (k-i) <= f_c):
      res += c_opt[k]*np.conj(c_opt[k-i])
  if (i == 0):
    res -= 1
  temp = res[0]
  u.append(temp)
roots = poly.polyroots(u) #find roots of the polynomial
circle_roots = []   #we filter out the roots that do not lie on the unit circle
for temp in roots:
  if(abs(1 - abs(temp)) < 0.01):
    circle_roots.append(temp)
roots_ordered = sorted(circle_roots, key=lambda x: x.real)   
single_roots = roots_ordered[0::2]   #according to theory, each root is a double root, so we throw away half of them
for temp in single_roots:   #extract locations of spikes from the phases of the roots
  p = cmath.phase(temp)
  if (p >=0):
    print(p/(2*np.pi))
  else:
    print(1+p/(2*np.pi))
