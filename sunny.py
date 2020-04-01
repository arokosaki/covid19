import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from wiki_scrape import from_wikipedia, chi_squared

gi = 0.065 # per day per person
gr = 1 / 28 # per day
eps = 0.01 # person
dt = 0.01 # day
T = 300 # day
population = 7_000_000

def plague(gi ,gr ,eps ,t , freq):
    it = t * freq
    p0 = 1-eps
    i0 = eps
    r0 = 0
    dt = 1 / freq
    time = np.arange(0,t,dt)
    p = np.zeros(it)
    i = np.zeros(it)
    r = np.zeros(it)
    p[0] = p0
    i[0] = i0
    r[0] = r0
    for n, t in enumerate(time[:-1]):
        p[n+1] = p[n] - dt * gi * p[n] * i[n]
        i[n+1] = i[n] + dt * gi * p[n] * i[n] - dt * gr * i[n]
        r[n+1] = r[n] + dt * gr * i[n]
    p = p[::freq]
    i = i[::freq]
    r = r[::freq]
    return p, i, r

df = from_wikipedia('israel')
def get_b(df):
    sigma = 1/(df['Total Cases'])**(1/2)
    df['Ln Total Cases'] = np.log(df['Total Cases'])
    x, y = df.index, df['Ln Total Cases']
    a , da, b, db, chi2  = chi_squared(x,y,sigma)
    return b, db

def chi_squared_non_linear(data, fit, sigma):
    chi = ((data-fit)/sigma)**2
    return chi.sum()

def fit(gi, eps, t):
    gr = 1 / 28  # per day
    freq = 100  # day
    _, i, r = plague(gi=gi,gr=gr,eps=eps,t=t,freq=freq)
    return i

def to_minimize(args):
    gi = args[0]
    ep = args[1]
    data = df['Total Cases']
    f = population * fit(gi, ep, data.size)
    sigma = data**(1/2)
    return chi_squared_non_linear(data, f, sigma)

if __name__ == '__main__':
    b, _ = get_b(df)
    res = minimize(to_minimize,np.array([b,1e-7]))
    print(res)
    d = df['Total Cases'].size
    print(d)
    p, i, r = plague(gr=gr,gi=res.x[0],eps=res.x[1],t=300,freq=100)
    # plt.scatter(y=df['Total Cases'],x=df.index)
    # plt.plot(population*(i+r))
    plt.plot(p)
    plt.plot(r)
    plt.plot(i)
    plt.show()