# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp



# ===== DAY 1 =====

## Zero Order Chemical Reactions
# Solve the differential equation associated with the zero order reaction
# Using the numerical approach

# Define the parameters and initial conditions
k = 5
initial_conditions = [1]
t_span = (0, 10)
t = np.linspace(*t_span, 11)

# Define the system of differential equations
def dAdt_fO(A, t):
    return k 

# Solve the system of differential equations
A = solve_ivp(dAdt_fO, t_span=t_span, t_eval=t, y0=initial_conditions)
A = A.y[0]

# Plot the solution
plt.figure(2)
plt.clf()
plt.xlim(0,10)
plt.ylim(0,max(A)+10)
plt.plot(t, A, linewidth=2)
plt.xlabel('time t')
plt.ylabel('Concentration A')
plt.title('Zero Order Reaction')
plt.legend()
plt.show()




# ---------- Prob. 1 ----------
# Plot the building blocks functions in Python for a few different parameter values.

x = np.arange(11)

# parameters
m = 10
b = 1
a = 1.5

y1 = m * x + b
y2 = a * np.exp(x)
y3 = 1 / (1 + x)

plt.figure(3)
plt.clf()
plt.subplot(1, 3, 1)
plt.plot(x, y1, 'red', linewidth=2)
plt.title('y = mx + b')
plt.subplot(1, 3, 2)
plt.plot(x, y2, 'blue', linewidth=2)
plt.title('y = ae^x')
plt.subplot(1, 3, 3)
plt.plot(x, y3, 'black', linewidth=2)
plt.title('y = 1/(1+x)')
plt.legend()
plt.show()




# ---------- Prob. 2 ----------
# Obtain a time course plot (in Python) for all the species in the following first-order chemical reactions:

# A -> 0
k = 5
A0 = [100]
t_span = (0, 10)
t = np.linspace(*t_span, 11)

# Define the system of differential equations
def dAdt_fO(A, t):
    return -k * A

# Solve the system of differential equations
A = solve_ivp(dAdt_fO, t_span=t_span, t_eval=t, y0=A0)
A = A.y[0]

# Plot the solution
plt.figure(2)
plt.clf()
plt.plot(t, A, linewidth=2)
plt.plot(t, np.zeros_like(t), '--k')
plt.xlabel('time t')
plt.ylabel('Concentration A')
plt.title('A -> 0')
plt.legend()
plt.show()



# A -> B
k = 5
initial_conditions = [0.5, 0]
t = np.arange(0, 11, 1)

# Define the system of differential equations
def systemOfDEs(y, t):
    A, B = y
    dAdt = -k * A
    dBdt = k * A
    return [dAdt, dBdt]

# Solve the system of differential equations
y = odeint(systemOfDEs, initial_conditions, t)

# Plot the solution
plt.plot(t, y[:, 0], '--r')
plt.plot(t, y[:, 1], '-k')
plt.xlabel('time t')
plt.ylabel('Concentrations')
plt.title('A -> B')
plt.legend(['A', 'B'])
plt.show()



# A -> B+C
k = 5
initial_conditions = [0.5, 0, 0]
t = np.arange(0, 11, 1)

# Define the system of differential equations
def systemOfDEs(y, t):
    A, B, C = y
    dAdt = -k * A
    dBdt = k * A
    dCdt = k * A
    return [dAdt, dBdt, dCdt]

# Solve the system of differential equations
y = odeint(systemOfDEs, initial_conditions, t)

# Plot the solution
plt.figure(2)
plt.clf()
plt.plot(t, y[:, 0], '--r')
plt.plot(t, y[:, 1], '-k')
plt.plot(t, y[:, 2], '-g')
plt.xlabel('time t')
plt.ylabel('Concentrations')
plt.title('A -> B+C')
plt.legend(['A', 'B', 'C'])
plt.show()




# ---------- Prob. 3 ----------
# Plot the solution for A, B, and C on the same plot with k=5, A0=B0=0.5, and C0=0.

# Define the parameter k and initial conditions
k = 5
initial_conditions = [0.5, 0.5, 0]
t = np.arange(0, 11, 1)

# Define the system of differential equations
def systemOfDEs(y, t):
    A = y[0]
    B = y[1]
    C = y[2]
    dAdt = -k * A * B
    dBdt = -k * A * B
    dCdt = k * A * B
    return [dAdt, dBdt, dCdt]

# Solve the system of differential equations
y = odeint(systemOfDEs, initial_conditions, t)

# Plot the solution
plt.figure(3)
plt.clf()
plt.plot(t, y[:, 0], '--r')
plt.plot(t, y[:, 1], '-k')
plt.plot(t, y[:, 2], '-b')
plt.xlabel('time t')
plt.ylabel('Concentrations')
plt.title('Second Order Reaction')
plt.legend(['A', 'B', 'C'])
plt.show()




# ---------- Prob. 4 ----------
# Find the steady state for the second order system. In Python, plot the time course and the steady state values. 
# Steady state at k=0, A=B=C=0

# Define the parameter k and initial conditions
k = 0
initial_conditions = [0, 0, 0]
t = np.arange(0, 11, 1)

# Define the system of differential equations
def systemOfDEs(y, t):
    A = y[0]
    B = y[1]
    C = y[2]
    dAdt = -k * A * B
    dBdt = -k * A * B
    dCdt = k * A * B
    return [dAdt, dBdt, dCdt]

# Solve the system of differential equations
y = odeint(systemOfDEs, initial_conditions, t)

# Plot the solution
plt.figure(3)
plt.clf()
plt.plot(t, y[:, 0], '--r')
plt.plot(t, y[:, 1], '-k')
plt.plot(t, y[:, 2], '-b')
plt.xlabel('time t')
plt.ylabel('Concentrations')
plt.title('Second Order Reaction')
plt.legend(['A', 'B', 'C'])
plt.show()

# Steady State for Y
all_Xs = np.arange(1, 11, 1)
all_Yss = []

for x in all_Xs:
    solution = odeint(systemOfDEs, initial_conditions, t)
    ss = solution[-1, 0]
    all_Yss.append(ss)

plt.figure(40)
plt.plot(all_Xs, all_Yss, '--b', linewidth=2)
plt.xlabel('Stimulus (X)')
plt.ylabel('Y Steady State')
plt.title('Gene Y regulated by tf X Steady State')
plt.legend()
plt.show()




# ---------- Prob. 5 ----------
# In the example for simple gene regulation, we have a combination of first and zero order
# reactions. Find the steady state of the system by hand. 
# What parameter(s) does the steady state depend on?

# Yss = p/r

"""
# For steady state
dY/dt = p-rY
0 = p-rYss
Yss = p/r 
"""




# ---------- Prob. 6 ----------
# Let’s look at the interconversion example. Plot a time course for both A & B. What do you see?

# Define the parameters and initial conditions
kf = 10
kr = 1
initial_conditions = [0.5, 0]
t = np.arange(0, 11, 1)

# Define the system of differential equations
def systemOfDEs(y, t):
    A, B = y
    dAdt = -kf * A + kr * B
    dBdt = kf * A - kr * B
    return [dAdt, dBdt]

# Solve the system of differential equations
y = odeint(systemOfDEs, initial_conditions, t)

# Plot the solution
plt.figure(3)
plt.clf()
plt.plot(t, y[:, 0], '--r', linewidth=2)
plt.plot(t, y[:, 1], '-k', linewidth=2)
plt.xlabel('time t')
plt.ylabel('Concentrations')
plt.title('Interconversion')
plt.legend(['A', 'B'])
plt.show()




# ---------- Prob. 7 ----------
# Take a look at the binding reaction below. What are the ODE’s for each species? What does a time course look like?

# Define the parameters and initial conditions
kon = 1
koff = 0.1
initial_conditions = [1, 0.5, 0]  # A, B, C

# Define the system of differential equations
def systemOfDEs(y, t):
    A, B, C = y
    dAdt = -kon * A * B + koff * C
    dBdt = -kon * A * B + koff * C
    dCdt = kon * A * B - koff * C
    return [dAdt, dBdt, dCdt]

# Solve the system of differential equations
y = odeint(systemOfDEs, initial_conditions, t)

# Plot the solution
plt.figure(3)
plt.clf()
plt.plot(t, y[:, 0], '--r', linewidth=2)
plt.plot(t, y[:, 1], '-k', linewidth=2)
plt.plot(t, y[:, 2], '-g', linewidth=2)
plt.xlabel('time t')
plt.ylabel('Concentrations')
plt.title('Binding Reaction')
plt.legend(['A', 'B', 'AB'])
plt.show()



# ==== Group problems ====
# ---------- Prob. 8 ----------
# Go back to problem 5. Play around with the parameters and plot various solutions.

plt.figure(8)
plt.clf()

# Define the parameters and initial conditions
p = 5
r = 1
initial_conditions = [0.5]
t = np.linspace(0, 10, 11)

# Define the system of differential equations
def systemOfDEs(y, t):
    return [p - r * y[0]]

# Solve the system of differential equations
y = odeint(systemOfDEs, initial_conditions, t)

# Plot the solution
plt.plot(t, y[:, 0], '--r', linewidth=2)
plt.plot([0, 15], [p / r, p / r], '--k', linewidth=2)
plt.xlabel('time t')
plt.ylabel('Concentrations')
plt.title('Simple gene transcription')
plt.legend(['Y'])
plt.ylim([0, 6])
plt.legend()
plt.show()




# ---------- Prob. 9a ----------
# Combine 2 first-order reactions to model this reversible reaction using Python.
# a. Does this reaction reach a steady-state?
# b. If it does reach a steady-state, which parameter determines how long it takes to reach steady state?

# Yes Yss, depends on kr and kf

plt.figure(9)
plt.clf()

# Define the parameters and initial conditions
kf = 5
kr = 1
initial_conditions = [0.5, 0]
t = np.arange(0, 11, 1)

# Define the system of differential equations
def systemOfDEs(y, t):
    A, B = y
    dAdt = -kf * A + kr * B
    dBdt = kf * A - kr * B
    return [dAdt, dBdt]

# Solve the system of differential equations
y = odeint(systemOfDEs, initial_conditions, t)

# Plot the solution
plt.plot(t, y[:, 0], '--r', linewidth=2)
plt.plot(t, y[:, 1], '--k', linewidth=2)
plt.xlabel('time t')
plt.ylabel('Concentrations')
plt.title('Simple Reversible')
plt.legend(['A', 'B'])
plt.show()



# ---------- Prob. 9b ----------
# Combine 2 first-order reactions to model the translocation of a protein from the
# Membrane to the Cytoplasm to the Nucleus.
# Plot all three time courses on the same graph, starting with 100% of the protein at the membrane.

plt.figure(9)
plt.clf()

# Define the parameters and initial conditions
k1 = 5
k2 = 1
initial_conditions = [1, 0, 0]
t = np.arange(0, 11, 1)

# Define the system of differential equationsns
def systemOfDEs(y, t):
    M, C, N = y
    dMdt = -k1 * M
    dCdt = k1 * M - k2 * C
    dNdt = k2 * C
    return [dMdt, dCdt, dNdt]

# Solve the system of differential equations
y = odeint(systemOfDEs, initial_conditions, t)

# Plot the solution
plt.plot(t, y[:, 0], '--r', linewidth=2)
plt.plot(t, y[:, 1], '--k', linewidth=2)
plt.plot(t, y[:, 2], '-b', linewidth=2)
plt.xlabel('time t')
plt.ylabel('Concentrations')
plt.title('Simple transport')
plt.legend(['M', 'C', 'N'])
plt.show()

