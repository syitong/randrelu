
# coding: utf-8

# In[2]:


from scipy.special import jv, gamma
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt


# In[24]:


d = 10.
R = lambda d: gamma(d/2+1)**(1/d)/np.sqrt(np.pi)
S = lambda d: 2*gamma(0.5)**(d+1)/gamma((d+1)/2)
# Radial density function
pr = lambda x: jv(d/2,2*np.pi*R(d)*x)**2*(R(d))**d*S(d-1)/x


# In[35]:


# According to Lemma 23 of Eldan and Shamir's 2016 paper to setup parameters
alpha = np.ceil(127*5/2/np.pi/d)
N = np.ceil(alpha**(3/2)*d**2)
# The support of target function is in [dom_scale, 2*dom_scale]
dom_scale = alpha * np.sqrt(d)
# Set the minimum gap in radius population for sampling according to pr. 
# It guarantees that every piece of target function, g_i, is supported by 
# N_supp points in the population.
N_supp = 100
min_gap = dom_scale / N / N_supp
R_grid = np.arange(min_gap, 3 * dom_scale, min_gap)
# Check the coverage of the domain of pr
integrate.quad(pr, min_gap, 3 * dom_scale)
# Sample 1/10 of the population of R according to pr
sample_size = int(len(R_grid) / 10)
p = pr(R_grid)
p = p / np.sum(p)
idx_sample = np.random.choice(len(R_grid),size=sample_size,p=p)
# Check the shape of the radius density
# fig = plt.hist(R_sample,bins=300,range=(0,5))
# plt.show()
# Generate the uniform direction samples
A_sample = np.random.randn(sample_size,d)
A_sample = A_sample / np.linalg.norm(A_sample, axis=1, keepdims=True)


# In[ ]:


# Construct smooth and non-smooth target functions
jr = lambda x: jv(d/2,2*np.pi*R(d)*x)**2*(R(d)/x)**d

idx = 0
g = []
l_cut = []
r_cut = []
while idx < len(R_grid):
    if idx < dom_scale / min_gap:
        g.append(0.)
        idx += 1
    else:
        if idx >= 2 * dom_scale / min_gap:
            g.append(1.)
            idx += 1
        else:
            criteria = (jr(R_grid[idx:idx + N_supp]) * R_grid[idx:idx + N_supp] 
                        > 1 / (80 * np.pi * R(d))
            if criteria.all():
                l_cut.append(idx)
                g.extend(np.ones(N_supp))
            else:
                g.extend(np.zeros(N_supp))
            idx += N_supp
g = np.array(g)
assert len(g) == len(R_grid)
                        
epsilon = np.random.choice([-1,1],len(l_cut))
for cut in l_cut:
    g[cut:cut+N_supp] *= epsilon[cut]

yg = g[idx_sample]
x_sample = A_sample * np.repeat(R_sample.reshape(-1,1), d, axis=1)


# In[95]:


# Smooth the value of g using smooth bump function (1/a)exp(-1/(1-(x/a)^2)
class bump:
    # create an array of the smooth bump function
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
    y_list = []
    a = int(mol.a / mol.min_gap)
    for idx in idx_list:
        left = max(idx-a,0)
        right = min(idx+a,len(g))
        y = np.sum(g[left:right] * mol.value[a+left-idx:a+right-idx]) * mol.min_gap
        y_list += [y]
    return np.array(y_list)

mol = bump(10 * min_gap, min_gap)
y_mol = mollify(idx_sample, g, mol)


# In[103]:


# Verify the bump and mollify
a = 0.1
x = np.arange(-2,2,0.001)
mol = bump(a, 0.001)
y = np.ones(len(x))
y[1300:1700] *= 0
y[3500:] *= 0
fig = plt.plot(x,y)
idx_list = np.arange(len(x))
idx_list_rand = np.random.choice(len(x), size=100)
y_mol = mollify(idx_list_rand,y,mol)

fig = plt.scatter(x[idx_list_rand],y_mol)
plt.show()

