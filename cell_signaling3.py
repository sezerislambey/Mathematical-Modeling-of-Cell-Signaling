# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint



# ===== DAY 3 =====
# ---------- Prob. 2 ---------- - Michaelis-Menten Kinetics

# Define the parameters and initial conditions
kon = 5
koff = 1
kcat = 1
initial_conditions = [10, 10, 0, 0]  # E, S, ES, P
t_span = (0, 50)
t = np.linspace(*t_span, 51)

# Define the system of differential equations
def system_of_des(y, t):
    E, S, ES, P = y
    dE_dt = -kon * E * S + koff * ES
    dS_dt = -kon * E * S + (koff + kcat) * ES
    dES_dt = kon * E * S - (koff + kcat) * ES
    dP_dt = kcat * ES
    return [dE_dt, dS_dt, dES_dt, dP_dt]

# Solve the system of differential equations
solution = odeint(system_of_des, initial_conditions, t)

# Plot the solution
plt.figure(2)
plt.plot(t, solution[:, 1], '-k', linewidth=2, label='S')
plt.plot(t, solution[:, 0], '--r', linewidth=2, label='E')
plt.plot(t, solution[:, 2], '-g', linewidth=2, label='ES')
plt.plot(t, solution[:, 3], '-b', linewidth=2, label='P')
plt.xlabel('time t')
plt.ylabel('Concentrations')
plt.title('MM Kinetics')
plt.legend()

# Plotting the MM equation
vMax = 5
kM = 1
Stot = np.arange(1, 101, 1)
Pvel = vMax * Stot / (kM + Stot)

plt.figure(20)
plt.plot(Stot, Pvel, '-k', linewidth=2)
plt.xlabel('S_tot')
plt.ylabel('P Velocity')
plt.title('MM Equation')
plt.legend(['P'])
plt.show()




# ---------- Prob. 3 ---------- - Hill Function with Different Values of n
# Define the parameters and initial conditions
k = 0.5
x_span = (0, 10)
x = np.linspace(*x_span, 101)

n = 1
y1 = (x ** n) / (x ** n + k ** n)

n = 2
y2 = (x ** n) / (x ** n + k ** n)

n = 5
y3 = (x ** n) / (x ** n + k ** n)

# Plot the solution
plt.figure(3)
plt.plot(x, y1, '-r', linewidth=2, label='n=1')
plt.plot(x, y2, '-b', linewidth=2, label='n=2')
plt.plot(x, y3, '-m', linewidth=2, label='n=5')
plt.xlabel('x')
plt.ylabel('Hill Function')
plt.title('Hill Function with Different Values of n')
plt.legend()
plt.show()




# ---------- Prob. 4 ---------- - Enzyme-catalyzed reaction with reverse reaction
# Go back to problem #2. Build a model of an enzyme-catalyzed reaction.
# Explore the effect of kon, koff, and kcat on the amount of ES complex formed and
# the rate of Product accumulation.

# Define the parameters and initial conditions
kon = 5
koff = 1
kcat = 1
initial_conditions = [0.5, 0.5, 0, 0]  # E, S, ES, P
t = np.arange(0, 11, 1)

# Define the system of differential equations
def system_of_de(y, t):
    dE_dt = -kon * y[0] * y[1] + koff * y[2]
    dS_dt = -kon * y[0] * y[1] + (koff + kcat) * y[2]
    dES_dt = kon * y[0] * y[1] - (koff + kcat) * y[2]
    dP_dt = kcat * y[2]
    return [dE_dt, dS_dt, dES_dt, dP_dt]

# Solve the system of differential equations
solution = odeint(system_of_de, initial_conditions, t)

# Plot the solution
plt.figure(4)
plt.plot(t, solution[:, 0], '--r', linewidth=2, label='E')
plt.plot(t, solution[:, 1], '-k', linewidth=2, label='S')
plt.plot(t, solution[:, 2], '-g', linewidth=2, label='ES')
plt.plot(t, solution[:, 3], '-b', linewidth=2, label='P')
plt.xlabel('time t')
plt.ylabel('Concentrations')
plt.title('Enzyme-Catalyzed Reaction with Reverse Reaction')
plt.legend()
plt.show()




# Add the simplest possible reverse reaction, P → S as a first order, uncatalyzed
# process, so that the system reaches a steady-state.

# Define the parameters and initial conditions
kon = 5
koff = 1
kcat = 1
kNew = 1
initial_conditions = [0.5, 0.5, 0, 0]  # E, S, ES, P
t = np.arange(0, 11, 1)

# Define the system of differential equations including the reverse reaction
def system_of_de(y, t):
    dE_dt = -kon * y[0] * y[1] + koff * y[2] + kNew * y[3]
    dS_dt = -kon * y[0] * y[1] + (koff + kcat) * y[2]
    dES_dt = kon * y[0] * y[1] - (koff + kcat) * y[2]
    dP_dt = kcat * y[2] - kNew * y[3]
    return [dE_dt, dS_dt, dES_dt, dP_dt]

# Solve the system of differential equations
solution = odeint(system_of_de, initial_conditions, t)

# Plot the solutions
plt.figure(5)
plt.plot(t, solution[:, 0], '--r', linewidth=2, label='E')
plt.plot(t, solution[:, 1], '-k', linewidth=2, label='S')
plt.plot(t, solution[:, 2], '-g', linewidth=2, label='ES')
plt.plot(t, solution[:, 3], '-b', linewidth=2, label='P')
plt.xlabel('time t')
plt.ylabel('Concentrations')
plt.title('MM Kinetics with Reverse Reaction')
plt.legend()
plt.show()




# ---------- Prob. 5 ----------
# Consider the 3-node gene regulatory system below:
# Plot a time course for this system with fixed parameters.


# Define the parameters and initial conditions
px = 1
py = 1
pz = 1
dx = 1
dy = 1
dz = 1
initial_conditions = [0.5, 0, 0]  # X, Y, Z
t = np.arange(0, 11, 1)

# Define the system of differential equations
def system_of_de(y, t):
    dX_dt = px - dx * y[0] * y[2]
    dY_dt = py * y[0] - dy * y[1]
    dZ_dt = pz * y[1] - dz * y[2]
    return [dX_dt, dY_dt, dZ_dt]

# Solve the system of differential equations
solution = odeint(system_of_de, initial_conditions, t)

# Plot the solutions
plt.figure(50)
plt.plot(t, solution[:, 0], '--r', linewidth=2, label='X')
plt.plot(t, solution[:, 1], '-k', linewidth=2, label='Y')
plt.plot(t, solution[:, 2], '-g', linewidth=2, label='Z')
plt.xlabel('time t')
plt.ylabel('Concentrations')
plt.title('3-Node Gene Regulatory System')
plt.legend()
plt.show()




# ---------- Prob. 6 ----------
# If we wanted to model a system that displays displays stable oscillations, that is similar
# to the one in problem #5, we could make one regulator more Hill-like. Consider the
# following ODEs:

# Solve this system and plot a time course. How does this system compare to the
# one from problem #5? You will need HIGH Hill numbers to see some interesting curves.

# Define the parameters and initial conditions
px = 3.62
py = 6.57
pz = 10
k = 1
dx = 0.35
dy = 0.8
dz = 1.06
n = 10
initial_conditions = [1, 0, 0]  # X, Y, Z
t = np.arange(0, 251, 1)

# Define the system of differential equations
def system_of_de(y, t):
    dX_dt = px - dx * y[0] * y[2]
    dY_dt = (py * (k ** n)) / ((k ** n) + (y[0] ** n)) - (dy * ((y[1] ** n)) / ((k ** n) + (y[1] ** n)))
    dZ_dt = (pz * (k ** n)) / ((k ** n) + (y[1] ** n)) - (dz * ((y[2] ** n)) / ((k ** n) + (y[2] ** n)))
    return [dX_dt, dY_dt, dZ_dt]

# Solve the system of differential equations
solution = odeint(system_of_de, initial_conditions, t)

# Plot the solutions
plt.figure(50)
plt.plot(t, solution[:, 0], '--r', linewidth=2, label='X')
plt.plot(t, solution[:, 1], '-k', linewidth=2, label='Y')
plt.plot(t, solution[:, 2], '-g', linewidth=2, label='Z')
plt.xlabel('time t')
plt.ylabel('Concentrations')
plt.title('3-Node Gene Regulatory System with Oscillations')
plt.legend()
plt.show()
