from scipy.special import jv, gamma
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class bump:
    # Create an array of the smooth bump function
    def __init__(self, a, min_gap):
        self.a = a
        self.min_gap = min_gap
        if a < min_gap:
            print("width parameter a cannot be smaller than the resolution.")
            return 0
        x = np.arange(-a,a,min_gap)
        y = 1/a * np.exp(-1 / (1+10**(-10) - (x/a)**2))
        normalizer = np.sum(y) * min_gap
        self.value = y / normalizer

def mollify(idx_list, g, mol):
    # Mollify a function via bump function
    y_list = []
    a = int(mol.a / mol.min_gap)
    for idx in idx_list:
        left = max(idx-a,0)
        right = min(idx+a,len(g))
        y = np.sum(g[left:right] * mol.value[a+left-idx:a+right-idx]) * mol.min_gap
        y_list += [y]
    return np.array(y_list)

def main(N_const=0.1, NSUPP=10000, mol_const=100, c_const=80, seed=0):
    np.random.seed(seed)
    d = 4
    R = lambda d: gamma(d/2+1)**(1/d)/np.sqrt(np.pi)
    S = lambda d: 2*gamma(0.5)**(d+1)/gamma((d+1)/2)
    # Radial density function
    pr = lambda x: jv(d/2,2*np.pi*R(d)*x)**2*(R(d))**d*S(d-1)/x

    # According to Lemma 23 of Eldan and Shamir's 2016 paper to setup parameters
    alpha = np.ceil(127*5/2/np.pi/d)
    # We can reduce N to obtain a less hard-to-learn target function
    N = np.ceil(N_const * alpha**(3/2)*d**2)

    # The support of target function is in [dom_scale, 2*dom_scale] in the
    # paper
    dom_scale = alpha * np.sqrt(d)

    # Set the minimum gap in radius population for sampling according to pr.
    # It guarantees that every piece of target function, g_i, is supported by
    # NSUPP points in the population.
    LBD = 0.2 * dom_scale
    RBD = 0.3 * dom_scale
    MINGAP = (RBD - LBD) / N / NSUPP
    R_grid = np.arange(LBD, RBD, MINGAP)
    offset = int(LBD / MINGAP)

    # Sample 1/10 of the population of R according to pr
    sample_size = int(len(R_grid) / 10)
    p = pr(R_grid)
    p = p / np.sum(p)
    idx_sample = np.random.choice(len(R_grid),size=sample_size,p=p)
    R_sample = R_grid[idx_sample]

    # Check the shape of the radius density
    # Check the coverage of the domain of pr
    mass ,err = integrate.quad(pr, LBD, RBD)
    fig = plt.figure()
    plt.plot(R_grid[::100],pr(R_grid[::100]) / mass)
    # plt.text(3.5,0.1,'(mass,err)=({0:.3f},{1:.3f})'.format(
    #    mass,err))
    plt.hist(R_sample,bins=300,density=True)
    plt.title("Histogram of Radius Density")
    plt.savefig('fig/hist.eps')

    # Generate the uniform direction samples
    A_sample = np.random.randn(sample_size,d)
    A_sample = A_sample / np.linalg.norm(A_sample, axis=1, keepdims=True)

    # Construct smooth and non-smooth target functions
    jr = lambda x: jv(d/2,2*np.pi*R(d)*x)**2

    idx = 0
    g = []
    l_cut = []
    r_cut = []
    while idx < len(R_grid):
        if idx + offset < LBD / MINGAP:
            g.append(0.)
            idx += 1
        else:
            if idx + offset >= RBD / MINGAP:
                g.append(0.)
                idx += 1
            else:
                # In the paper the constant on the denominator is chosen to be
                # 80. It is related to the support of the target function.
                # Here we adjust it according to LBD.
                criteria = (jr(R_grid[idx:idx + NSUPP]) * R_grid[idx:idx + NSUPP]
                            > 1 / (c_const * np.pi * R(d)))
                if criteria.all():
                    l_cut.append(idx)
                    g.extend(np.ones(min(NSUPP,len(R_grid)-idx)))
                else:
                    g.extend(np.zeros(min(NSUPP,len(R_grid)-idx)))
                idx += NSUPP
    g = np.array(g)
    assert len(g) == len(R_grid)

    epsilon = np.random.choice([-1,1],len(l_cut))
    for idx, cut in enumerate(l_cut):
        g[cut:min(cut+NSUPP,len(g))] *= epsilon[idx]

    yg = g[idx_sample]
    x_sample = A_sample * np.repeat(R_sample.reshape(-1,1), d, axis=1)

    # Smooth the value of g using smooth bump function (1/a)exp(-1/(1-(x/a)^2)
    mol = bump(mol_const * MINGAP, MINGAP)
    y_mol = mollify(idx_sample, g, mol)

    # Plot yg and y_mol
    sort_idx = np.argsort(R_sample)
    fig = plt.figure()
    plt.scatter(R_sample[sort_idx[::10]],yg[sort_idx[::10]],c='r')
    plt.plot(R_sample[sort_idx[::10]],y_mol[sort_idx[::10]],c='b')
    plt.title("g and smoothed g")
    plt.savefig('fig/gplot.eps')

    # Save dataset
    print(x_sample.shape)
    # with open('data/eldan-train-data.npy','bw') as f:
    #     np.save(f, x_sample[:int(0.8*len(x_sample))])
    # with open('data/eldan-train-label.npy', 'bw') as f:
    #     np.save(f, yg[:int(0.8*len(x_sample))])
    # with open('data/eldan-test-data.npy', 'bw') as f:
    #     np.save(f, x_sample[int(0.8*len(x_sample)):])
    # with open('data/eldan-test-label.npy', 'bw') as f:
    #     np.save(f, yg[int(0.8*len(x_sample)):])
    #
    # with open('data/eldan-smooth-train-data.npy','bw') as f:
    #     np.save(f, x_sample[:int(0.8*len(x_sample))])
    # with open('data/eldan-smooth-train-label.npy', 'bw') as f:
    #     np.save(f, y_mol[:int(0.8*len(x_sample))])
    # with open('data/eldan-smooth-test-data.npy', 'bw') as f:
    #     np.save(f, x_sample[int(0.8*len(x_sample)):])
    # with open('data/eldan-smooth-test-label.npy', 'bw') as f:
    #     np.save(f, y_mol[int(0.8*len(x_sample)):])

if __name__ == '__main__':
    main(N_const=0.01,NSUPP=100000,mol_const=10000)
