# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp



# ===== DAY 2 =====

# ---------- Prob. 2 ---------- - Positive Autoregulation

# Define the parameters and initial conditions
k1 = 0.1
k2 = 0.5
initial_conditions = [1, 0, 0]
t_span = (0, 20)
t = np.linspace(*t_span, 21)

# Define the system of differential equations
def system_of_des(y, t):
    A, B, C = y
    dA_dt = -k1 * A
    dB_dt = k1 * A - k2 * B
    dC_dt = k2 * B
    return [dA_dt, dB_dt, dC_dt]

# Solve the system of differential equations
solution = odeint(system_of_des, initial_conditions, t)

# Plot the solution
plt.figure(2)
plt.plot(t, solution[:, 0], '--r', linewidth=2, label='MAPKKK')
plt.plot(t, solution[:, 1], '--b', linewidth=2, label='MAPKK')
plt.plot(t, solution[:, 2], '--k', linewidth=2, label='MAPK')
plt.xlabel('time t')
plt.ylabel('Concentrations')
plt.title('Positive Autoregulation')
plt.legend()
plt.show()


# ---------- Prob. 1 ---------- - Positive and Negative Autoregulation
# Let’s get Python to solve these autoregulation schemes and plot a time course for each species.

# Define the parameters and initial conditions
p = 5
k = 1
r = 1
initial_conditions = [0.5]
t_span = (0, 10)
t = np.linspace(*t_span, 11)

# Positive Autoregulation
# Define the system of differential equations
def system_of_des_positive(y, t):
    Y = y[0]
    dY_dt = p * Y / (k + Y) - r * Y
    return [dY_dt]

# Solve the system of differential equations
solution = odeint(system_of_des_positive, initial_conditions, t)

# Plot the solution:
plt.figure(3)
plt.plot(t, solution[:, 0], '--r', linewidth=2)
plt.xlabel('time t')
plt.ylabel('Concentrations')
plt.title('Positive Autoregulation (Y)')
plt.legend()
plt.show()



# Negative Autoregulation
# Define the system of differential equations
p = 5
k = 1
r = 1
initial_conditions = [0.5]
t_span = (0, 10)
t = np.linspace(*t_span, 11)

# Define the system of differential equations
def system_of_des_negative(y, t):
    Y = y[0]
    dY_dt = p * k / (k + Y) - r * Y
    return [dY_dt]

# Solve the system of differential equations
solution = odeint(system_of_des_negative, initial_conditions, t)

# Plot the solution
plt.figure(4)
plt.plot(t, solution[:, 0], '--b', linewidth=2)
plt.xlabel('time t')
plt.ylabel('Concentrations')
plt.title('Negative Autoregulation (Y)')
plt.legend()
plt.show()




# ==== Group problems ====
# ---------- Prob. 4 ---------- - Gene Y regulated by tf X 
# Build a model of a gene Y that is regulated by transcription factor X
# Treat X as a constant, not a variable and solve for Y in Python.

# Define the parameters and initial conditions
pMax = 10
k = 1
r = 1
n = 1
x = 5
initial_conditions = [0.5]
t_span = (0, 10)
t = np.linspace(*t_span, 11)

# Define the system of differential equations
def system_of_des_gene_regulated(y, t):
    Y = y[0]
    dY_dt = pMax * (x**n) / (k**n + x**n) - r * Y
    return [dY_dt]

# Solve the system of differential equations
solution = odeint(system_of_des_gene_regulated, initial_conditions, t)

# Plot the solution
plt.figure(5)
plt.plot(t, solution[:, 0], '--r', linewidth=2)
plt.title('Gene Y regulated by tf X')

# Steady State for Y
all_Xs = np.arange(1, 11, 1)
all_Yss = []

for x in all_Xs:
    solution = odeint(system_of_des_gene_regulated, initial_conditions, t)
    ss = solution[-1, 0]
    all_Yss.append(ss)

plt.figure(40)
plt.plot(all_Xs, all_Yss, '--b', linewidth=2)
plt.xlabel('Stimulus (X)')
plt.ylabel('Y Steady State')
plt.title('Gene Y regulated by tf X Steady State')
plt.legend()




# =============================================================================
# ---------- Prob. 5 ---------- - Positive Cascades
# Model the positive cascades.
# Obtain plots that look like this (show all 3 time courses on the same plot). Will need HIGH Hill numbers.
# What does your plot look like if all Hill numbers are 1? To do a valid comparison,
# get your graphs in 9B to reach ~ the same steady state as in 9A. In the graphs
# above, nothing happens for the first minute. You don’t need to do this; your
# simulation (and graphs) can begin where X starts rising.

# Define the parameters and initial conditions
xMax = 1
yMax = 1
zMax = 1
ny = 10
nz = 10
ky = 1
kz = 5
r = 1
x = 5
initial_conditions = [0.5, 0, 0]
t_span = (0, 20)
t = np.linspace(*t_span, 21)

# Define the system of differential equations
def system_of_des_positive_cascades(y, t):
    X, Y, Z = y
    dX_dt = xMax - r * X
    dY_dt = yMax * (x**ny) / (ky**ny + x**ny) - r * Y
    dZ_dt = zMax * (x**nz) / (kz**nz + x**nz) - r * Z
    return [dX_dt, dY_dt, dZ_dt]

# Solve the system of differential equations
solution = odeint(system_of_des_positive_cascades, initial_conditions, t)

# Plot the solution
plt.figure(6)
plt.plot(t, solution[:, 0], '--r', linewidth=2, label='X')
plt.plot(t, solution[:, 1], '--b', linewidth=2, label='Y')
plt.plot(t, solution[:, 2], '--k', linewidth=2, label='Z')
plt.xlabel('time t')
plt.ylabel('Concentrations')
plt.title('Positive Cascades')
plt.legend()
plt.show()
