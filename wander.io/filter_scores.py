import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def gen_syn_loc():
    random.seed(123)
    xys_list = []
    #n_gauss1, n_gauss2 = 15,10 
    n_gauss1, n_gauss2 = 50, 50 
    #mu1x, mu1y, mu2x, mu2y = 10, 10, 12, 14
    mu1x, mu1y, mu2x, mu2y = 0, -10, 12, 8
    #sigma1x, sigma1y, sigma2x, sigma2y = 5, 6, 7, 7
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

def plot(xys_list, ax, k=None):
    color = 'grey'
    color_sel = 'red'
    if k is None:
        x_list = [xys[0] for xys in xys_list]
        y_list = [xys[1] for xys in xys_list]
        s_list = [xys[2] for xys in xys_list]
        plot_helper(x_list, y_list, s_list, ax, color)
    else:
        s_list = [xys[2] for xys in xys_list]
        s_list_topk = (np.array(s_list).argsort()[-k:][::-1]).tolist()
        x_list1, y_list1, s_list1 = [], [], []
        x_list2, y_list2, s_list2 = [], [], []
        for i, xys in enumerate(xys_list):
            x,y,s = xys
            if i in s_list_topk:
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

def filter_xys_list(xys_list, max_itr, num_neighbors, sensitivity, bias):
    s_list = [xys[2] for xys in xys_list]
    S = np.array(s_list)

    L = get_dists(xys_list)
    l_avg = np.mean(L)

    for _ in range(max_itr):
        S_neighbors = (l_avg-L)*np.exp(S)
        #S_neighbors_filtered = S_neighbors.argsort(axis = 0)
        S_neighbors_filtered = np.sort(S_neighbors, axis=0)[-num_neighbors:][::-1]
        S_agg = np.sum(S_neighbors_filtered, axis=0)
        S_tilde = S_agg + bias*np.exp(S)
        S = 10*sigmoid(sensitivity*S_tilde/np.max(S_tilde))
    
    xys_filtered_list = []
    for i, s in enumerate(S):
        x,y,_ = xys_list[i]
        xys_filtered_list.append((x,y,s))
    return xys_filtered_list

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

def filter_xys_list(xys_list, max_itr, num_neighbors, sensitivity, bias):
    s_list = [xys[2] for xys in xys_list]
    S = np.array(s_list)

    L = get_dists(xys_list)
    l_avg = np.mean(L)

    for _ in range(max_itr):
        S_neighbors = (l_avg-L)*np.exp(S)
        S_neighbors_filtered = np.sort(S_neighbors, axis=0)[-num_neighbors:]
        S_agg = np.sum(S_neighbors_filtered, axis=0)
        S_tilde = S_agg + bias*np.exp(S)
        S = 10*sigmoid(sensitivity*S_tilde/np.max(S_tilde))
    
    xys_filtered_list = []
    for i, s in enumerate(S):
        x,y,_ = xys_list[i]
        xys_filtered_list.append((x,y,s))
    return xys_filtered_list

def get_cluster_id_list(xys_list, num_clusters):
    xy_list = [(xys[0], xys[1]) for xys in xys_list]
    xy_list = np.array(xy_list)
    kmeans = KMeans(n_clusters=num_clusters).fit(xy_list)
    centroids = kmeans.cluster_centers_ 
    cluster_id_list = kmeans.labels_ 
    return cluster_id_list, centroids

def cluster_xys_list(cluster_id_list, xys_list, num_clusters):
    xys_list_clusters = [[] for _ in range(num_clusters)]
    for i, xys in enumerate(xys_list):
        k = int(cluster_id_list[i])
        xys_list_clusters[k].append(xys)
    return xys_list_clusters

def main():
    xys_list = gen_syn_loc()
    num_clusters = 2
    cluster_id_list, centroids = get_cluster_id_list(xys_list, num_clusters)
    xys_list_clusters = cluster_xys_list(cluster_id_list, xys_list, num_clusters)

    xys_list_temp = xys_list.copy()
    xys_list_temp.extend([(centroid[0], centroid[1], float('inf')) for centroid in centroids])
    ax = plt.gca()
    ax.set_title('Input Scores and Cluster Centroids')
    ax.set_xlabel('X location')
    ax.set_ylabel('Y location')
    plot(xys_list_temp, ax, k=num_clusters)
    plt.show()

    for i, xys_list in enumerate(xys_list_clusters):
        xys_filtered_list = filter_xys_list(xys_list, max_itr=3, num_neighbors=2, sensitivity=5, bias=50)

        ax1 = plt.subplot(211)
        ax1.set_title('Input Scores for Cluster {}'.format(i))
        ax1.set_xlabel('X location')
        ax1.set_ylabel('Y location')
        plot(xys_list, ax1)

        k = 5
        ax2 = plt.subplot(212)
        ax2.set_title('Filtered Scores (topk={}) for Cluster {}'.format(k, i))
        ax2.set_xlabel('X location')
        ax2.set_ylabel('Y location')
        plot(xys_filtered_list, ax2, k=k)
        plt.show()

if __name__ == '__main__':
    main()
