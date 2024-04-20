import numpy as np

# Define each benchmark function in Python
def F1(x):
    # Bent Cigar (Unimodal)
    # D = x.shape[1]
    z = x[0]**2 + 10**6*np.sum(x[1:]**2)
    return z

def F2(x):
    # Power (Unimodal)
    return np.sum(np.abs(x)**(np.arange(len(x)) + 2))

def F3(x):
    # Zakharov (Unimodal)
    s = 0.5 * np.sum(x)
    return np.sum(x**2) + s**2 + s**4

def F4(x):
    # Rosenbrock (Unimodal)
    return np.sum(100 * (x[:-1]**2 - x[1:])**2 + (x[:-1] - 1)**2)

def F5(x):
    return 10**6 * x[0]**2 + np.sum(x[1:]**2)

def F6(x):
    D = len(x)
    return np.sum(((10**6)**((np.arange(D))/(D-1))) * x**2)

def F7(x):
    terms = np.sin(np.sqrt(x**2 + np.roll(x, -1)**2))**2 - 0.5
    denom = 1 + 0.001 * (x**2 + np.roll(x, -1)**2)**2
    return 0.5 + np.sum(terms / denom)

def F8(x):
    w = 1 + (x - 1)/4
    terms = (w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2)
    return np.sin(np.pi * w[0])**2 + np.sum(terms) + (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)

def F9(x):
    y = x + 4.209687462275036e+002
    mask1 = np.abs(y) < 500
    mask2 = y > 500
    mask3 = y < -500
    f = np.zeros_like(y)
    f[mask1] = y[mask1] * np.sin(np.sqrt(np.abs(y[mask1])))
    f[mask2] = (500 - y[mask2] % 500) * np.sin(np.sqrt(500 - y[mask2] % 500)) - (y[mask2] - 500)**2 / (10000 * len(x))
    f[mask3] = (y[mask3] % 500 - 500) * np.sin(np.sqrt(y[mask3] % 500 - 500)) - (y[mask3] + 500)**2 / (10000 * len(x))
    return 418.9829 * len(x) - np.sum(f)

def F10(x):
    D = len(x)
    return -20 * np.exp(-0.2 * np.sqrt((1/D) * np.sum(x**2))) - np.exp((1/D) * np.sum(np.cos(2 * np.pi * x))) + 20 + np.exp(1)

def F11(x):
    x = x + 0.5
    a, b = 0.5, 3
    kmax = 20
    k = np.arange(kmax + 1)
    c1 = a**k
    c2 = 2 * np.pi * b**k
    f = np.sum(c1 * np.cos(c2 * x[:, np.newaxis]), axis=1)
    return f - np.sum(c1 * np.cos(c2 * 0.5))

def F12(x):
    D = len(x)
    return (np.abs(np.sum(x**2) - D)**(1/4)) + (0.5 * np.sum(x**2) + np.sum(x))/D + 0.5

def F13(x):
    dim = len(x)
    return np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, dim+1)))) + 1

def F14(x):
    dim = len(x)
    term1 = 10 * (np.sin(np.pi * (1 + (x[0] + 1)/4)))**2
    sum_term = np.sum((((x[:-1] + 1)/4)**2) * (1 + 10 * (np.sin(np.pi * (1 + (x[1:] + 1)/4)))**2))
    term2 = ((x[-1] + 1)/4)**2
    return (np.pi/dim) * (term1 + sum_term + term2) + np.sum(Ufun(x, 10, 100, 4))

def Ufun(x, a, k, m):
    return k * (((x - a)**m) * (x > a) + ((-x - a)**m) * (x < -a))

# Define a dictionary for easy function mapping
function_map = {
    'F1': (F1, -100, 100, 30),
    'F2': (F2, -100, 100, 30),
    'F3': (F3, -100, 100, 30),
    'F4': (F4, -100, 100, 30),
    'F5': (F5, -100, 100, 30),
    'F6': (F6, -100, 100, 30),
    'F7': (F7, -100, 100, 30),
    'F8': (F8, -100, 100, 30),
    'F9': (F9, -100, 100, 30),
    'F10': (F10, -32.768, 32.768, 30),
    'F11': (F11, -100, 100, 30),
    'F12': (F12, -100, 100, 30),
    'F13': (F13, -600, 600, 30),
    'F14': (F14, -50, 50, 30),
}

def BenchmarkFunctions(F):
    fobj, lb, ub, dim = function_map[F]
    return lb, ub, dim, fobj
