# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from numpy.linalg import inv
from numpy import sqrt
import matplotlib.pyplot as plt

#Déclaration des variables
N = 5
dt = 0.25
Deff = 10**(-10)

S = 10**(-8)
R = 1
Ce = 10
h = R/(N-1)

# Matrices pour résolution cas stationnaire avec N = 5
A = np.array((
[-1 , 1             , 0           , 0               ], 
[1  , -(2 + (N-1)*h/R), 1 + 4*h/R   , 0               ], 
[0  , 1             , -(2+2*h/R), 1 + 2*h/R       ],
[0  , 0             , 1           , -(2 + 4*h/3/R)]))

B = np.array((
    [0                           ],
    [S*h*h/Deff                  ],
    [S*h*h/Deff                  ],
    [-Ce*(1 + 4*h/3/R)+S*h*h/Deff]
))


def Code_Calcul(L, n_t, i):
    
    for j in range (n_t):
        M = [] #On créer une matrice M pour calculer les valeurs aux différents noeuds au pas de temps suivants.
        M.append(0)
        for k in range (1, N-1):
            M.append( L[k] + dt*(Deff/h/h*(L[k+1] - 2*L[k]+ L[k-1] + (N-1)*h/k/R*(L[k+1] - L[k])) - S))
        M.append(Ce)    
        L = M    
        L[0] = L[1] # On a la condition initiale de la dérivée nul en 0. Donc C(T + dT, 0) = C(T + DT, R/4)
    return L[i]


#Cas stationnaire pour N = 5
def Code_stationnaire(i): 
    C = np.dot(inv(A), B) #Calcul de la solution
    D = C.tolist()
    L = []
    for k in range(len(C)):
        L.append(D[k][0]) #On construit une liste pour accéder aux valeur
    L.append(Ce) #à l'extrémité, on a c(t, R) = Ce
    return L[i]
    
def Solution_analytique(x):
    
    return (S/4/Deff)*R*R*(x**2/R/R - 1) + Ce

## Implémentation des erreurs pour N = 5


def Erreur_L1(L):
    C = np.dot(inv(A), B)
    D = C.tolist()
    D.append([Ce])
    s = 0
    for k in range(len(L)):
        s+= 1/5*abs(D[k][0] - Solution_analytique(k*R/(N-1)))
    return s

def Erreur_L2(L):
    C = np.dot(inv(A), B)
    D = C.tolist()
    D.append([Ce])
    s = 0
    for k in range(len(L)):
        s+= 1/5*abs(D[k][0] - Solution_analytique(k*R/(N-1)))**2
    return sqrt(s)


def Erreur_Linf(L):
    C = np.dot(inv(A), B)
    D = C.tolist()
    D.append([Ce])
    E = []
    for k in range (len(L)):
        E.append(abs(D[k][0] - Solution_analytique(k*R/(N-1))))
    return max(E)

# Affichage des différentes courbes

def affichage(L):
    X = []
    Y = []
    Z = []
    for k in range(len(L)):
        Z.append(Code_stationnaire(k))
        Y.append(Solution_analytique(k*R/(N-1)))
        X.append(k/(len(L)-1))
    plt.plot(X, Z)
    plt.plot(X, Y)
    plt.show()















#Matrices pour résolutions cas stationnaire avec deuxième méthode, avec N = 5

A_2 = np.array((
[-1        , 1             , 0           , 0       ], 
[1 - 2*h/R , -2            , 1 + 2*h/R   , 0       ], 
[0         , 1 -h/R        , -2          , 1 + h/R ],
[0         ,  0            , 1 -2*h/3/R  , -2      ]))

B_2 = np.array((
    [0                           ],
    [S*h*h/Deff                  ],
    [S*h*h/Deff                  ],
    [-Ce*(1 + 2*h/3/R)+S*h*h/Deff]
))


## Avec la nouvelle technique de différenciation


def Code_stationnaire_bis(i):
    C = np.dot(inv(A_2), B_2)
    D = C.tolist()
    L = []
    for k in range(len(C)):
        L.append(D[k][0])
    L.append(Ce)
    return L[i]

#Affichage des 3 courbes
def affichage_bis(L):
    X = []
    Y = []
    Z = []
    W = []
    for k in range(len(L)):
        W.append(Code_stationnaire(k))
        Z.append(Code_stationnaire_bis(k))
        Y.append(Solution_analytique(k*R/(N-1)))
        X.append(k/(len(L)-1))
    plt.plot(X, Z)
    plt.plot(X, Y)
    plt.plot(X, W)
    plt.show()

#Implémentation des erreurs

def Erreur_L1_bis(L):
    C = np.dot(inv(A_2), B_2)
    D = C.tolist()
    D.append([Ce])
    s = 0
    for k in range(len(L)):
        s+= 1/5*abs(D[k][0] - Solution_analytique(k*R/(N-1)))
    return s

def Erreur_L2_bis(L):
    C = np.dot(inv(A_2), B_2)
    D = C.tolist()
    D.append([Ce])
    s = 0
    for k in range(len(L)):
        s+= 1/5*abs(D[k][0] - Solution_analytique(k*R/(N-1)))**2
    return sqrt(s)


def Erreur_Linf_bis(L):
    C = np.dot(inv(A_2), B_2)
    D = C.tolist()
    D.append([Ce])
    E = []
    for k in range (len(L)):
        E.append(abs(D[k][0] - Solution_analytique(k*R/(N-1))))
    return max(E)











#Cas général premier cas de différenciation===============
##========================================================
def Code_stationnaire_general1(n):
    a = np.zeros((n, n))
    b = np.zeros((n, 1))
    h1 = R/(n-1)
    a[0][0] = -1
    a[0][1] = 1
    a[n-1][n-1] = 1
    b[0][0] = 0
    b[n-1][0] = Ce
    for k  in range (1, n-1):
        a[k][k-1] = 1
        a[k][k] = -(2+ (n-1)*h1/k/R)
        a[k][k+1] =(1+ (n-1)*h1/k/R)
        b[k][0] = h1*h1*S/Deff 
    C = np.dot(inv(a), b)
    D = C.tolist()
   
    return D

def Erreur_L1_general1(n):
    D = Code_stationnaire_general1(n)
    s = 0
    for k in range(n):
        s+= 1/n*abs(D[k][0] - Solution_analytique(k*R/(n-1)))
    return s

def Erreur_L2_general1(n):
    D = Code_stationnaire_general1(n)
    s = 0
    for k in range(n):
        s+= 1/n*abs(D[k][0] - Solution_analytique(k*R/(n-1)))**2
    return sqrt(s)


def Erreur_Linf_general1(n):
    D = Code_stationnaire_general1(n)
    E = []
    for k in range (n):
        E.append(abs(D[k][0] - Solution_analytique(k*R/(n-1))))
    return max(E)


def affichage_generale1(n):
    X = np.linspace(0, 1, n)
    Y = Code_stationnaire_general1(n)
    Z = []
    for k in range(n):
        Z.append(Solution_analytique(k*R/(n-1)))
    plt.plot(X, Y)
    plt.plot(X, Z)
    plt.show()

#Cas général deuxième cas de différenciation =============
##========================================================

def Code_stationnaire_general2(n):
    a = np.zeros((n, n))
    b = np.zeros((n, 1))
    a[0][0] = -1
    a[0][1] = 1
    a[n-1][n-1] = 1
    b[0][0] = 0
    b[n-1][0] = Ce
    h1 = R/(n-1)
    for k  in range (1, n-1):
        a[k][k-1] = 1-h1/R*(n-1)/k/2
        a[k][k] = -2
        a[k][k+1] =1+h1/R*(n-1)/k/2
        b[k][0] = h1*h1*S/Deff 
    C = np.dot(inv(a), b)
    D = C.tolist()
   
    return D


def affichage_generale2(n):
    X = np.linspace(0, 1, n)
    Y = Code_stationnaire_general2(n)
    Z = []
    for k in range(n):
        Z.append(Solution_analytique(k*R/(n-1)))
    plt.plot(X, Y)
    plt.plot(X, Z)
    plt.show()


def Erreur_L1_general2(n):
    D = Code_stationnaire_general2(n)
    s = 0
    for k in range(n):
        s+= 1/n*abs(D[k][0] - Solution_analytique(k*R/(n-1)))
    return s

def Erreur_L2_general2(n):
    D = Code_stationnaire_general2(n)
    s = 0
    for k in range(n):
        s+= 1/n*abs(D[k][0] - Solution_analytique(k*R/(n-1)))**2
    return sqrt(s)


def Erreur_Linf_general2(n):
    D = Code_stationnaire_general2(n)
    E = []
    for k in range (n):
        E.append(abs(D[k][0] - Solution_analytique(k*R/(n-1))))
    return max(E)



    