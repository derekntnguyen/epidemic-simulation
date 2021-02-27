# Epidemic Simulation

This python script simulates both a deterministic model and a stochastic model of the spread of an epidemic and compares the two results. The simulation can be completed with different types of individuals and different number of vaccines.

The deterministic model of the epidemic simulation utilizes solve_ivp with equations given in dndt and jacobian given in function jacobian to simulate the epidemic.

The stochastic model of the edpidemic simulation utilizes the gillespie algorith to simulate the epidemic.
