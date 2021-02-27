#  -*- coding: utf-8 -*-
'''
Elba Midterm Project
Derek Nguyen
Created: 2020-04-28
Modified: 2020-04-28
'''

# %% codecell

import numpy as np
import scipy.integrate as integrate
import matplotlib # used to create interactive plots in the Hydrogen package of the Atom IDE
matplotlib.use('Qt5Agg') # used to create interactive plots in the Hydrogen package of the Atom IDE
import matplotlib.pyplot as plt

def ode_model(t, n0):
    '''
    Deterministic model of Elba Epidemic utilizing solve_ivp with equations given in
    dndt and jacobian given in function jacobian
    Parameters:
    t, an array of values in days
    n0, initial conditions
    '''
    k = [1.76e-5, 0.88e-5, 0.100, 0.010, 0.030, 3.52e-6] # Rates for persons

    def dndt(t, n): # The differential equation matrix used in solve_ivp
        nHY, nHYF, nHE, nHEF, nSY, nSE, nI, nD, nV = n #unpack the tuple
        dn = np.array((-k[0] * nHY * nSY - k[5] * nHY * nV, # Healthy Young
                       -k[0] * nHYF * nSY, # Healthy Young Freeloaders
                       -k[1] * nHE * nSE - k[5] * nHE * nV, # Healthy Elder
                       -k[1] * nHEF * nSE, # Healthy Elder Freeloaders
                       k[0] * nHY * nSY + k[0] * nHYF * nSY - k[2] * nSY - k[3] * nSY, # Sick Young
                       k[1] * nHE * nSE + k[1] * nHEF * nSE - k[2] * nSE - k[4] * nSE, # Sick Elder
                       k[2] * nSY + k[2] * nSE + k[5] * nHY * nV + k[5] * nHE * nV, # Immune
                       k[3] * nSY + k[4] * nSE, # Dead
                       -k[5] * nHY * nV - k[5] * nHE * nV)) # Vaccine
        return dn

    def jacobian(t, n): # The jacobian calculated by hand, utilized in solve_ivp
        nHY, nHYF, nHE, nHEF, nSY, nSE, nI, nD, nV = n #unpack the tuple
        dfdy = np.array([[-k[0] * nSY - k[5] * nV, 0, 0, 0, -k[0] * nHY, 0, 0, 0, - k[5] * nHY], # Healthy Young
                         [0, -k[0] * nSY, 0, 0, -k[0] * nHYF, 0, 0, 0, 0], # Healthy Young Freeloaders
                         [0, 0, -k[1] * nSE - k[5] * nV, 0, 0, -k[1] * nHE, 0, 0, - k[5] * nHE], # Healthy Elders
                         [0, 0, 0, -k[1] * nSE, 0, -k[1] * nHEF, 0, 0, 0], # Healthy Elder Freeloaders
                         [k[0] * nSY,  k[0] * nSY, 0, 0, k[0] * nHY + k[0] * nHYF - k[2] - k[3], 0, 0, 0, 0], # Sick Young
                         [0, 0, k[1] * nSE, k[1] * nSE, 0, k[1] * nHE + k[1] * nHEF - k[2] - k[4], 0, 0, 0], # Sick Elders
                         [k[5] * nV, 0, k[5] * nV, 0, k[2], k[2], 0, 0, k[5] * nHY + k[5] * nHE], # Immune
                         [0, 0, 0, 0, k[3], k[4], 0, 0, 0], # Dead
                         [-k[5] * nV, 0, k[5] * nV, 0, 0, 0, 0, 0 , -k[5] * nHY - k[5] * nHE]]) # Vaccine
        return dfdy

    sol = integrate.solve_ivp(fun = dndt, t_span = [np.min(t), np.max(t)], y0 = n0, t_eval = t, method = 'Radau', jac = jacobian)

    return sol

def gillespie_model(timeSpan, nMax, n0):
    '''
    Stochastic model of Elba epidemic utilizng the gillespie algorithm
    Parameters:
    timeSpan, an integer in days
    nMax, number of iterations to perform
    n0, initial conditions
    '''
    k = [1.76e-5, 0.88e-5, 0.100, 0.010, 0.030, 3.52e-6] # Rates for persons in days^-1
    t_final = timeSpan
    ii = 0 # create counter to be used in while loop

    v = np.array(([-1, 0, 0, 0, 1, 0, 0, 0, 0], # establish v matrix used for modifiying the counts of each individual within the while loop
                  [0, -1, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, -1, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, -1, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, -1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, -1, 1, 0, 0],
                  [0, 0, 0, 0, -1, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, -1, 0, 1, 0],
                  [-1, 0, 0, 0, 0, 0, 1, 0, -1],
                  [0, 0, -1, 0, 0, 0, 1, 0, -1]))

    nHY, nHYF, nHE, nHEF, nSY, nSE, nI, nD, nV = n0 # unpack the tuple
    n0 = [nHY, nHYF, nHE, nHEF, nSY, nSE, nI, nD, nV] # rearrange in list for indexing

    t = np.zeros((nMax, 1)) # preallocate t and n
    n = np.zeros((nMax, 9))
    n[0] = n0
    n[29][8] = nV / 2 # assign distribution of vaccines at day 30

    while ii <= nMax: # while loop to iterate continuously through nMax iterations
        ii += 1 # reaction counter

        if t[ii] >= t_final: # breaks the loop of time exceeds timeSpan
            t = t[:ii]
            n = n[:ii,:]
            break

        nHY, nHYF, nHE, nHEF, nSY, nSE, nI, nD, nV = n[ii-1] # set initial conditions within n

        r_s = np.array((k[0] * nHY * nSY, # establish rs for which r_total can be calculated for
                        k[0] * nHYF * nSY,
                        k[1] * nHE * nSE,
                        k[1] * nHEF * nSE,
                        k[2] * nSY,
                        k[2] * nSE,
                        k[3] * nSY,
                        k[4] * nSE,
                        k[5] * nHY * nV,
                        k[5] * nHE * nV))

        r_total = r_s[0] + r_s[1] + r_s[2] + r_s[3] + r_s[4] + r_s[5] + r_s[6] + r_s[7] + r_s[8] + r_s[9]

        if r_total == 0: # break if event total = 0
            t = t[:ii]
            n = n[:ii,:]
            break

        dt = -np.log(np.random.uniform(0, 1)) / r_total # determine time step for which the model will progress
        t[ii] = t[ii-1] + dt

        p = r_s / r_total # calculate probability of events
        csp = np.cumsum(p) # utilize cumsum function and where function to determine which event to perform
        q = np.random.uniform()
        kk = np.where(q < csp)[0][0]

        n[ii, :] = n[ii-1, :] + v[kk, :] # modifiy data depending on which event was performed and utilizing v

    return t,n

def elba(nY0 = (27199,0,1), nE0 = (4800,0,0), nV0 = 0, timeSpan = 120, nMax = 2000000, nRun = 1):
    ############################## Deterministic Section ###########################################
    t_initial = np.linspace(0, 30, 500) # break up timeSpan into 2 seperate groups such that vaccines can be distributed at day 30
    t_final = np.linspace(30, timeSpan, 500)

    nHY, nHYF, nSY = nY0 # unpacking tuples for initial young person conditions
    nHE, nHEF, nSE = nE0 # unpacking tuples for initial elder person conditions

    n0 = [nHY, nHYF, nHE, nHEF, nSY, nSE, 0, 0, nV0] # establish initial conditions

    osol_i = ode_model(t_initial, n0) #call deterministic model utilizing the first timeSpan

    n0_final = (osol_i.y[0][-1], osol_i.y[1][-1], osol_i.y[2][-1], osol_i.y[3][-1], osol_i.y[4][-1], osol_i.y[5][-1], osol_i.y[6][-1], osol_i.y[7][-1], nV0/2) # reset initial conditions for second timeSpan based on the final values of timeSpan 1

    osol_f = ode_model(t_final, n0_final) # call deterministic model again utilizing new variables after vaccine distribution

    o_HYHYF_initial = osol_i.y[0] + osol_i.y[1] # add Healthy and Freeloaders into 1 array
    o_HEHEF_initial = osol_i.y[2] + osol_i.y[3]
    o_HYHYF_final = osol_f.y[0] + osol_f.y[1]
    o_HEHEF_final = osol_f.y[2] + osol_f.y[3]

    o_nDead = osol_f.y[7][-1] # establish death toll

    ################################# Stochastic Section ####################################
    nDead = np.zeros(nRun) # preallocate array for storing number of dead used in histogram
    runs = np.arange(nRun)

    for iterations in runs: # utilize a for loop to iterate through number of nRuns
        t, n = gillespie_model(timeSpan, nMax, n0)
        g_HYHYF = n[:,0] + n[:,1]
        g_HEHEF = n[:,2] + n[:,3]
        g_nDead = n[-1,7]
        nDead[iterations] = g_nDead

    ################################## Plots ###############################################
    plt.subplot(1,2,1) # plots 6 individual lines representing all individuals

    oaxes = plt.gca()
    oaxes.set_xlim([0, timeSpan])
    oaxes.set_ylim([-1000, 35000])

    plt.plot(osol_i.t, o_HYHYF_initial.T, 'blue')
    plt.plot(osol_i.t, o_HEHEF_initial.T, 'orange')
    plt.plot(osol_i.t, osol_i.y[4].T, 'green')
    plt.plot(osol_i.t, osol_i.y[5].T, 'red')
    plt.plot(osol_i.t, osol_i.y[6].T, 'purple')
    plt.plot(osol_i.t, osol_i.y[7].T, 'brown')

    plt.plot(osol_f.t, o_HYHYF_final.T, 'blue')
    plt.plot(osol_f.t, o_HEHEF_final.T, 'orange')
    plt.plot(osol_f.t, osol_f.y[4].T, 'green')
    plt.plot(osol_f.t, osol_f.y[5].T, 'red')
    plt.plot(osol_f.t, osol_f.y[6].T, 'purple')
    plt.plot(osol_f.t, osol_f.y[7].T, 'brown')

    plt.title('Elba Epidemic Deterministic Model')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of People')
    plt.legend(('HY + HYF', 'HE + HEF', 'SY', 'SE', 'I', 'D'), loc = 'best')
    plt.text(timeSpan-30, o_nDead+1800, 'Death Toll =' + str(int(o_nDead)))

    if nRun == 1: # creates a graph of 6 individual lines representing all individuals if nRun = 1
        plt.subplot(1,2,2)

        gaxes = plt.gca()
        gaxes.set_xlim([0, timeSpan])
        gaxes.set_ylim([-1000, 35000])

        plt.plot(t, g_HYHYF, 'blue')
        plt.plot(t, g_HEHEF, 'orange')
        plt.plot(t, n[:,4], 'green')
        plt.plot(t, n[:,5], 'red')
        plt.plot(t, n[:,6], 'purple')
        plt.plot(t, n[:,7], 'brown')


        plt.title('Elba Epidemic Stochastic Model')
        plt.xlabel('Time (days)')
        plt.ylabel('Number of People')
        plt.legend(('HY + HYF', 'HE + HEF', 'SY', 'SE', 'I', 'D'), loc = 'best')
        plt.text(timeSpan-30, g_nDead+1800, 'Death Toll =' + str(int(g_nDead)))

    else: # creates a histogram of Runs v number of Dead if nRun > 1
        plt.subplot(1,2,2)
        plt.hist(nDead)

        plt.title('Elba Epidemic Stochastic Model')
        plt.xlabel('Number of Deaths')
        plt.ylabel('Number of Runs')
