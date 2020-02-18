import random
import numpy as np
import matplotlib.pyplot as plt

def gen_syn_loc():
    random.seed(123)
    xys_list = []
    #n_gauss1, n_gauss2 = 30,20 
    n_gauss1, n_gauss2 = 15,10 
    mu1x, mu1y, mu2x, mu2y = 10, 10, 12, 14
    sigma1x, sigma1y, sigma2x, sigma2y = 5, 6, 7, 7
    for _ in range(n_gauss1):
        x = random.gauss(mu1x, sigma1x)
        y = random.gauss(mu1y, sigma1y)
        s = 10*random.random()
        xys_list.append((x,y,s))
    for _ in range(n_gauss2):
        x = random.gauss(mu2x, sigma2x)
        y = random.gauss(mu2y, sigma2y)
        s = 10*random.random()
        xys_list.append((x,y,s))
    random.shuffle(xys_list)
    return xys_list

def plot(xys_list, ax, thresh=None):
    color = 'grey'
    color_sel = 'red'
    if thresh is None:
        x_list = [xys[0] for xys in xys_list]
        y_list = [xys[1] for xys in xys_list]
        s_list = [xys[2] for xys in xys_list]
        plot_helper(x_list, y_list, s_list, ax, color)
    else:
        x_list1, y_list1, s_list1 = [], [], []
        x_list2, y_list2, s_list2 = [], [], []
        for xys in xys_list:
            x,y,s = xys
            if s > thresh:
                x_list2.append(x)
                y_list2.append(y)
                s_list2.append(s)
            else:
                x_list1.append(x)
                y_list1.append(y)
                s_list1.append(s)
        plot_helper(x_list1, y_list1, s_list1, ax, color)
        plot_helper(x_list2, y_list2, s_list2, ax, color_sel)


def plot_helper(x_list, y_list, s_list, ax, color):
    ax.scatter(x_list, y_list, c=color)
    for i, s in enumerate(s_list):
        ax.annotate('{0:.2f}'.format(s), (x_list[i], y_list[i]))

def get_dists(xys_list):
    N = len(xys_list)
    L = np.zeros(shape=(N,N)) 
    xy_list = [(xys[0],xys[1]) for xys in xys_list]
    for i in range(N):
        for j in range(i, N):
            coord1 = np.array(xy_list[i])
            coord2 = np.array(xy_list[j])
            l = np.linalg.norm(coord1-coord2) 
            L[i][j] = l
            L[j][i] = l
    return L 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def filter_xys_list(xys_list, max_itr, agg_type='sum', sensitivity=10, bias=100):
    s_list = [xys[2] for xys in xys_list]
    S = np.array(s_list)

    L = get_dists(xys_list)
    l_avg = np.mean(L)

    if agg_type == 'sum':
        AGG = np.sum
    elif agg_type == 'max':
        AGG = np.max
    else:
        assert False
    for _ in range(max_itr):
        S_temp = AGG((l_avg-L)*np.exp(S), axis=0)+bias*np.exp(S)
        S = 10*sigmoid(sensitivity*S_temp/np.max(S_temp))
    
    xys_filtered_list = []
    for i, s in enumerate(S):
        x,y,_ = xys_list[i]
        xys_filtered_list.append((x,y,s))
    return xys_filtered_list

def main():
    xys_list = gen_syn_loc()
    xys_filtered_list = filter_xys_list(xys_list, 5000)

    ax1 = plt.subplot(211)
    ax1.set_title('Input Scores')
    ax1.set_xlabel('X location')
    ax1.set_ylabel('Y location')
    plot(xys_list, ax1)

    ax2 = plt.subplot(212)
    ax2.set_title('Filtered Scores, Thresh=8.0')
    ax2.set_xlabel('X location')
    ax2.set_ylabel('Y location')
    plot(xys_filtered_list, ax2, thresh=8.0)
    plt.show()

if __name__ == '__main__':
    main()
