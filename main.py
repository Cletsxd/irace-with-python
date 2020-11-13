import argparse
import logging
import sys

from pylab import *
import time
import math
import random as rnd
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

def classifier_accuracy_per_fold(features, x_train, y_train, x_test, y_test):
    features_index = np.asarray(np.where(features == 1)[0])

    # New sets for new features
    x_train_new = x_train.iloc[:, features_index]
    x_test_new = x_test.iloc[:, features_index]
    
    #print(x_train_new.shape, y_train.shape, x_test_new.shape, y_test.shape)
    #print(x_train.shape, y_train.shape, x_test_newshape, y_test)
    
    # Training classifier
    #clf = GaussianNB()
    #clf.fit(x_train_new, y_train)
    
    #clf = KNeighborsClassifier(n_neighbors=3)
    #clf.fit(x_train_new, y_train)
    
    clf = svm.SVC()
    clf.fit(x_train_new, y_train)
    
    # Testing classifier
    y_pred = clf.predict(x_test_new)
    
    return accuracy_score(y_test, y_pred)

# Here computes the accuracy for some classifier
# The classifier may change
def classifier_accuracy(features, X, Y, kf):
    features_index = np.asarray(np.where(features == 1)[0])
    
    Xnew = X.iloc[:, features_index]
    
    # New sets for new features
    #x_train_new = x_train.iloc[:, features_index]
    #x_test_new = x_test.iloc[:, features_index]
    
    #print(x_train_new.shape, y_train.shape, x_test_new.shape, y_test.shape)
    #print(x_train.shape, y_train.shape, x_test_newshape, y_test)
    
    # Training classifier
    #clf = GaussianNB()
    #clf.fit(x_train_new, y_train)
    
    #clf = KNeighborsClassifier(n_neighbors=3)
    #clf.fit(x_train_new, y_train)
    
    clf = svm.SVC()
    #clf.fit(x_train_new, y_train)
    fold_acc = np.zeros(10)
    #total_acc = np.zeros(10)
    #for fold in range(10):
    
    i = 0
    for train_index, test_index in kf.split(Xnew, Y):
        x_train, x_test = Xnew.iloc[train_index], Xnew.iloc[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        fold_acc[i] = accuracy_score(y_test, y_pred)
        
        i += 1
        
    #total_acc[fold] = fold_acc.mean()
    
    return fold_acc.mean()        
    
    # Testing classifier
    #y_pred = clf.predict(x_test_new)
    
    #return accuracy_score(y_test, y_pred)

# Compute the number of 1s in a feature vector
def R(X):
    return sum(X)

# Compute fitness mohamad2011
# X: feature vector
# TG: total of features
# w_1: priority weight
def fitness(X, w_1, TG, XF, Y, kf):
    w_2 = 1 - w_1
    
    accuracy = classifier_accuracy(X, XF, Y, kf)
    
    return (w_1 * accuracy) + ((w_2 * (TG - R(X))) / TG), accuracy

def norm(value, max_val, min_val):
    return (value - min_val)/(max_val - min_val)

def fitness_norm(X, w_1, TG, XF, Y, kf):
    w_2 = 1 - w_1
    
    accuracy = classifier_accuracy(X, XF, Y, kf)
    
    R_x = norm(R(X), TG, 0)
    
    return (w_1 * accuracy) + (w_2 * (1 - R_x)), accuracy

def fitness_no(X, w_1, TG, XF, Y, kf):
    return fitness_norm(X, w_1, TG, XF, Y, kf)

def probe_fitness(accuracy, features):
    return (0.8 * accuracy) + ((0.2 * (22282 - features)) / 22282)

def to_binary(X):
    binary = []
    
    for x_i in X:
        if x_i >= 0.5:
            binary.append(1)
        else:
            binary.append(0)
    
    return np.asarray(binary)

def initialize_pop(X, Y, kf, N, atts, w_1, ll, uu, min_vel, max_vel):
    final_pop = []
    
    for _ in range(N):
        variable_vector = []
        velocity_vector = []
        
        for _ in range(atts):
            variable_vector.append(rnd.uniform(ll, uu))
            velocity_vector.append(rnd.uniform(min_vel, max_vel))
        
        variable_vector = np.asarray(variable_vector)
        bin_variable_vector = to_binary(variable_vector)
        
        velocity_vector = np.asarray(velocity_vector)
        
        fit, accuracy = fitness(bin_variable_vector,
                                w_1,
                                atts,
                                X, Y, kf)
        
        final_pop.append([variable_vector,
                          bin_variable_vector,
                          velocity_vector,
                          fit,
                          accuracy,
                          np.array([variable_vector, bin_variable_vector, fit, accuracy])])
    
    return np.asarray(final_pop)

def update_generation_best(pop):
    best_i = pop[0]
    best_fit = pop[0][4]
    
    # First individual
    for ind in pop:
        if ind[4] > best_fit:
            best_fit = ind[4]
            best_i = ind
    
    return best_i

def velocity_update(ind, w, c_1, c_2, real_best, K = 1):
    # Individual (ind) data
    variable_vector = ind[0]
    velocity_vector = ind[2]
    p_best = ind[5][0]
    
    # Best individual data
    b_var_vec = real_best[0]
    
    new_velocity_vector = []
    
    for i in range(len(velocity_vector)):
        r_1 = rnd.uniform(0, 1)
        r_2 = rnd.uniform(0, 1)
        
        new_v_i = (velocity_vector[i] * w) + (c_1 * r_1 * (p_best[i] - variable_vector[i])) + (c_2 * r_2 * (b_var_vec[i] - variable_vector[i]))
        
        #new_v_i = K * (velocity_vector[i] + (c_1 * r_1 * (p_best[i] - variable_vector[i])) + (c_2 * r_2 * (b_var_vec[i] - variable_vector[i])))
        
        #new_v_i = deterministic_back(new_v_i)
        
        new_velocity_vector.append(new_v_i)
    
    return np.array([variable_vector,
                     ind[1],
                     np.asarray(new_velocity_vector),
                     ind[3],
                     ind[4],
                     ind[5]])

def reflection(x_i, l, u):
    out_bounds = False
    
    x = x_i
    
    while x < l or x > u:
        if x < l:
            x = (2 * l) - x
            out_bounds = True
        elif x > u:
            x = (2 * u) - x
            out_bounds = True
    
    return x, out_bounds

def deterministic_back(vel):
    return -1 * (0.5 * vel)

def feature_update(ind, X, Y, kf, w_1, num_atts, low, upp, phi, phi_2):
    variable_vector = ind[0]
    velocity_vector = ind[2]
    
    new_variable_vector = variable_vector + velocity_vector
    
    for i in range(len(new_variable_vector)):
        new_variable_vector[i], bounds = reflection(new_variable_vector[i], low, upp)
        
        if bounds:
            velocity_vector[i] = deterministic_back(velocity_vector[i])
    
    new_bin_feature_vector = to_binary(new_variable_vector)
    
    # Mutation
    # phi decreases for each generation
    for i in range(len(new_variable_vector)):
        if new_bin_feature_vector[i] == 1:
            if rnd.uniform(0, 1) > phi:
               new_bin_feature_vector[i] = 0
               new_variable_vector[i] = rnd.uniform(0, 0.5)
    
    # Mutation method for 0 features
    while(sum(new_bin_feature_vector) == 0):
        for i in range(len(new_variable_vector)):
            if rnd.uniform(0, 1) < phi_2:
                new_variable_vector[i] = rnd.uniform(0.5, 1)
                new_bin_feature_vector = 1
    
    fit, accuracy = fitness(new_bin_feature_vector, w_1, num_atts, X, Y, kf)
    
    return np.array([new_variable_vector,
                     new_bin_feature_vector,
                     velocity_vector,
                     fit,
                     accuracy,
                     ind[5]])

def p_best_update(ind):
    # Individual current data
    variable_vec = ind[0]
    bin_var_vec = ind[1]
    velocity_vec = ind[2]
    curr_fit = ind[3]
    curr_acc = ind[4]
    
    # Individual best data
    best_var_vec = ind[5][0]
    best_bin_var_vec = ind[5][1]
    best_fit = ind[5][2]
    best_acc = ind[5][3]
    
    if curr_fit >= best_fit:
        return np.array([variable_vec,
                         bin_var_vec,
                         velocity_vec,
                         curr_fit,
                         curr_acc,
                         np.array([variable_vec, bin_var_vec, curr_fit, curr_acc])])
    else:
        return np.array([variable_vec,
                         bin_var_vec,
                         velocity_vec,
                         curr_fit,
                         curr_acc,
                         np.array([best_var_vec, best_bin_var_vec, best_fit, best_acc])])

def update_global_best(generation_best, real_global_best):
    # Generation best individual data
    gen_fit = generation_best[3]
    
    # Real global best individual data
    real_fit = real_global_best[3]
    
    if gen_fit > real_fit:
        return np.array([generation_best[0],
                         generation_best[1],
                         generation_best[2],
                         generation_best[3],
                         generation_best[4],
                         generation_best[5]])
    else:
        return np.array([real_global_best[0],
                         real_global_best[1],
                         real_global_best[2],
                         real_global_best[3],
                         real_global_best[4],
                         real_global_best[5]])

def phi_update(phi, max_gen, plowl, pfil):
    if phi > plowl:
        return phi - (1 / max_gen)
    else:
        return rnd.uniform(pfil, plowl)

def main(POPULATION, C_1, C_2, W, W_1, INITIAL_PHI, PHI_UPPER_LIMIT, PHI_LOWER_LIMIT, VEL_UPP, VEL_LOW, DATFILE):
    df = pd.read_csv("../parkinson_smoted.csv")
    num_atts = df.shape[1] - 1 # Number of attributes
    num_rows = df.shape[0] # Number of cases

    # Best seed based on previous experiments
    rnd.seed(1)
    np.random.seed(1)

    X = df.iloc[:, 0:22282]
    Y = df["target"]

    kf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 0)

    Y_int = []
    
    for i in range(len(Y)):
        if Y[i] == "Parkinson's Disease":
            Y_int.append(0)
        elif Y[i] == "Neurological Disease Control":
            Y_int.append(1)
        elif Y[i] == "Healthy Control":
            Y_int.append(2)

    Y = np.asarray(Y_int)

    #### PARAMETERS ####
    # Lower limit for variables vector
    lol = 0.0

    # Upper limit for variables vector
    upl = 1.0

    # Generations
    g = 35

    # Population
    N = POPULATION

    # Learning coefficients
    c_1 = C_1
    c_2 = C_2

    # Intertia
    w = W # Inertia

    # Priority weight from the fitness function
    w_1 = W_1
    
    # Probability mutate data
    phi = INITIAL_PHI
    phi_lower_limit = PHI_UPPER_LIMIT
    phi_first_limit = PHI_LOWER_LIMIT

    # Velocity data
    max_vel = VEL_UPP
    min_vel = VEL_LOW

    population = initialize_pop(X, Y, kf, N, num_atts, w_1, lol, upl, max_vel, min_vel)

    real_best = update_generation_best(population)

    for gen in range(0, g):
        for i in range(N):
            # Velocity update
            ind1 = velocity_update(population[i], w, c_1, c_2, real_best)
                
            # Feature update
            ind2 = feature_update(ind1, X, Y, kf, w_1, num_atts, lol, upl, phi, phi_lower_limit, max_vel, min_vel)
                
            # Personal best update
            new_ind = p_best_update(ind2)
            
            # Individual update
            population[i] = new_ind

        gen_best = update_generation_best(population)

        real_best = update_global_best(gen_best, real_best)

        phi = phi_update(phi, g, phi_lower_limit, phi_first_limit)

    # save the fo values in DATFILE
    with open(DATFILE, 'w') as f:
        f.write(str(real_best[3] * 100))

    with open("results.txt", "w") as f:
        f.write(" *** Last individual *** \n" +
                " > Number of features:" + str(sum(real_best[1])) + "\n" +
                " > Accuracy:", str(real_best[4]) + "\n" +
                " > Fitness:", str(real_best[3]))

if __name__ == "__main__":
    # just check if args are ok
    with open('args.txt', 'w') as f:
        exe = str(sys.argv[0])
        instance = str(sys.argv[3]).split("/")[-1]

        rest_params = sys.argv[4:14]

        to_write = [exe, instance]
        for param in rest_params:
            to_write.append(param)

        last_param = str(sys.argv[-2] + "="+ sys.argv[-1])
        to_write.append(last_param)

        f.write(str(to_write))

    # Params [2, 11]
    pop = int(str(to_write[2]).split("=")[1])
    c1 = float(str(to_write[3]).split("=")[1])
    c2 = float(str(to_write[4]).split("=")[1])
    w = float(str(to_write[5]).split("=")[1])
    w1 = float(str(to_write[6]).split("=")[1])
    phi = float(str(to_write[7]).split("=")[1])
    phiup = float(str(to_write[8]).split("=")[1])
    philo = float(str(to_write[9]).split("=")[1])
    velup = float(str(to_write[10]).split("=")[1])
    vello = float(str(to_write[11]).split("=")[1])

    # .dat file
    dat = str(to_write[12]).split("=")[1]

    main(pop, c1, c2, w, w1, phi, phiup, philo, velup, vello, dat)