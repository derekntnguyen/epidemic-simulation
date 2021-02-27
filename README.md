# Epidemic Simulation

## Summary
This python script simulates both a deterministic model and a stochastic model of the spread of an epidemic and compares the two results. 

The first method utilized was a deterministic method utilizing differential equations and the solve_ivp function within the SciPy package. This method provided a continuous data set with which the epidemic could be modeled. 

The second method was a stochastic method utilizing the Gillespie Algorithm. The second method utilized random number generation and weighted events to create a model that accounted for random chance. The deterministic model also has the option to provide a histogram of the number of runs of the simulation against the number of deaths. This gives us the ability to see the general accuracy of the deterministic model by being able to compare the data across multiple iterations of the model.

## Results
![image](https://user-images.githubusercontent.com/79728577/109375805-91657780-788d-11eb-90a2-aaf53df90fe7.png)

Figure 1: 15,000 Vaccines and 0 Non-Vaccinators

![image](https://user-images.githubusercontent.com/79728577/109375813-9fb39380-788d-11eb-8068-b67e9884fe46.png)

Figure 2: Young Non-Vaccinators
