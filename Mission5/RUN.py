import numpy as np
from RungeKutta import RungeKutta
from initialization import initialization

def RUN(nP, MaxIt, lb, ub, dim, fobj):
    Cost = np.zeros(nP)
    X = initialization(nP, dim, ub, lb)  
    Xnew2 = np.zeros(dim)
    
    Convergence_curve = np.zeros(MaxIt)
    
    for i in range(nP):
        Cost[i] = fobj(X[i, :])

    Best_Cost = np.min(Cost)    # Determine the Best Solution
    ind = np.argmin(Cost)
    Best_X = X[ind, :]

    Convergence_curve[0] = Best_Cost
    
    it = 1
    while it < MaxIt:
        f = 20 * np.exp(-(12 * (it / MaxIt)))
        Xavg = np.mean(X, axis=0)
        SF = 2 * (0.5 - np.random.rand(nP)) * f
        
        for i in range(nP):
            ind_l = np.argmin(Cost)
            lBest = X[ind_l, :]
            
            A, B, C = RndX(nP, i)
            ind1 = np.argmin(Cost[[A, B, C]])
            
            gama = np.random.rand() * (X[i, :] - np.random.rand(dim) * (ub - lb)) * np.exp(-4 * it / MaxIt)
            Stp = np.random.rand(dim) * ((Best_X - np.random.rand() * Xavg) + gama)
            DelX = 2 * np.random.rand(dim) * (np.abs(Stp))
            
            if Cost[i] < Cost[ind1]:
                Xb = X[i, :]
                Xw = X[ind1, :]
            else:
                Xb = X[ind1, :]
                Xw = X[i, :]
            
            # SM = RungeKutta(Xb, Xw, DelX)  # Define 'RungeKutta'
            SM = RungeKutta(Xb, Xw, DelX)

            L = np.random.rand(dim) < 0.5
            Xc = L * X[i, :] + (1 - L) * X[A, :]
            Xm = L * Best_X + (1 - L) * lBest
            
            vec = [1, -1]
            flag = np.floor(2 * np.random.rand(dim) + 1).astype(int)
            r = np.array([vec[f-1] for f in flag])
            
            g = 2 * np.random.rand()
            mu = 0.5 + 0.1 * np.random.randn(dim)
            
            if np.random.rand() < 0.5:
                Xnew = (Xc + r * SF[i] * g * Xc) + SF[i] * SM + mu * (Xm - Xc)
            else:
                Xnew = (Xm + r * SF[i] * g * Xm) + SF[i] * SM + mu * (X[A, :] - X[B, :])

            # Ensure all solutions stay within bounds
            Xnew = np.clip(Xnew, lb, ub)
            CostNew = fobj(Xnew)

            if CostNew < Cost[i]:
                X[i, :] = Xnew
                Cost[i] = CostNew
            
            # ESQ Implementation
            if np.random.rand() < 0.5:
                EXP = np.exp(-5 * np.random.rand() * it / MaxIt)
                # r = np.floor(np.random.uniform(-1, 2))
                r = np.floor(Unifrnd(-1, 2, 1, 1))

                u = 2 * np.random.rand(dim)
                # w = np.random.uniform(0, 2, dim) * EXP
                w = Unifrnd(0, 2, 1, dim)[0] * EXP
                
                A, B, C = RndX(nP, i)
                Xavg = (X[A, :] + X[B, :] + X[C, :]) / 3
                
                beta = np.random.rand(dim)
                Xnew1 = beta * Best_X + (1 - beta) * Xavg
                
                for j in range(dim):
                    if w[j] < 1:
                        Xnew2[j] = Xnew1[j] + r * w[j] * abs((Xnew1[j] - Xavg[j]) + np.random.randn())
                    else:
                        Xnew2[j] = (Xnew1[j] - Xavg[j]) + r * w[j] * abs((u[j] * Xnew1[j] - Xavg[j]) + np.random.randn())

                Xnew2 = np.clip(Xnew2, lb, ub)
                CostNew = fobj(Xnew2)
                
                if CostNew < Cost[i]:
                    X[i, :] = Xnew2
                    Cost[i] = CostNew
                
                elif np.random.rand() < w[np.random.randint(dim)]:
                    SM = RungeKutta(X[i, :], Xnew2, DelX)
                    Xnew = (Xnew2 - np.random.rand() * Xnew2) + SF[i] * (SM + (2 * np.random.rand(dim) * Best_X - Xnew2))
                    
                    Xnew = np.clip(Xnew, lb, ub)
                    CostNew = fobj(Xnew)
                    
                    if CostNew < Cost[i]:
                        X[i, :] = Xnew
                        Cost[i] = CostNew

        if Cost[i] < Best_Cost:
            Best_X = X[i, :]
            Best_Cost = Cost[i]

        Convergence_curve[it] = Best_Cost
        print(f'it : {it}, Best Cost = {Convergence_curve[it]}')
        it += 1

    return Best_Cost, Best_X, Convergence_curve

def Unifrnd(a, b, c, dim):
    a2 = a / 2
    b2 = b / 2
    mu = a2 + b2
    sig = b2 - a2
    z =  mu + sig * (2 * np.random.rand(c, dim) - 1)
    return z

def RndX(nP, i):
    Qi = np.random.permutation(nP)
    Qi = Qi[Qi != i]
    A = Qi[0]
    B = Qi[1]
    C = Qi[2]
    return A, B, C

