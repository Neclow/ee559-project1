import dlc_practical_prologue as prologue
import torch
import matplotlib.pyplot as plt


def standardize(train, test):
    m, s = train.mean(), train.std()
    return (train-m)/s, (test-m)/s


def load_data(N=1000):
    trainX, trainY, trainC, testX, testY, testC = prologue.generate_pair_sets(N)

    trainX, testX = standardize(trainX, testX)

    return trainX, trainY, trainC, testX, testY, testC

def get_params(L, N, D, T):
    # Solve: N = (L-1)*H^2 + (D+T+L)*H+T    
    a = L-1
    b = D+T+L
    c = T-N

    s = pow(b**2-4*a*c, 0.5)
    return round((-b+s)/(2*a))

def width_visualization(N=70000, D=14*14, T=10):
    fig, ax = plt.subplots(1,1)

    h = []
    x = range(2,31)

    for L in x:
        h.append(get_params(L, N, D, T))

    plt.plot(x,h)
    plt.ylabel('width')
    plt.xlabel('depth')
    plt.show()