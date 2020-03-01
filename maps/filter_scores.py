import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import namedtuple
import json

Attraction = namedtuple('Attraction', ['latlon', 'score', 'is_hotel', 'is_restaurant', 'metadata_google'])

#####################################################################
# K-means Clustering
#####################################################################
def get_cluster_id_list(attraction_list, num_clusters):
    # use k-means clustering
    xy_list = [attraction.latlon for attraction in attraction_list]
    xy_list = np.array(xy_list)
    kmeans = KMeans(n_clusters=num_clusters).fit(xy_list)
    centroids = kmeans.cluster_centers_ 
    cluster_id_list = kmeans.labels_ 
    return cluster_id_list, centroids

def cluster_attraction_list(cluster_id_list, attraction_list, num_clusters):
    # seperate xys_list by clusters
    attraction_list_clusters = [[] for _ in range(num_clusters)]
    for i, attraction in enumerate(attraction_list):
        k = int(cluster_id_list[i])
        attraction_list_clusters[k].append(attraction)
    return attraction_list_clusters

#####################################################################
# Score Filtering/Curation 
#####################################################################
def select_locales(S, metadata):
    num_neighbors, = metadata
    S_idxs = np.argsort(S, axis=0)[-num_neighbors:]
    return S_idxs

def filter_attraction_list(attraction_list, max_itr, num_neighbors, sensitivity, bias):
    # get scores
    s_list = [attraction.score for attraction in attraction_list]
    S = np.array(s_list)
    S_original = np.array(s_list)

    # get distance matrix
    L = get_dists(attraction_list)
    l_avg = np.mean(L)

    metadata = (num_neighbors,)
    for _ in range(max_itr):
        S_idxs = select_locales(S, metadata)
        S_neighbors = ((l_avg-L)*S)[S_idxs]
        S_agg = np.sum(S_neighbors, axis=0)
        S_tilde = S_agg + bias*S_original*S_original
        #S = 10*sigmoid(sensitivity*S_tilde/np.max(S_tilde))
        S = 10*S_tilde/np.max(S_tilde)
        '''
        # iteratively refine scores
        # collect neighbor scores
        S_avg = np.mean(S)
        S_neighbors = (l_avg-L)*S# - L*(S-S_avg)
        # aggregate to top num_neighbor neighbors with the highest scores
        S_neighbors_filtered = np.sort(S_neighbors, axis=0)[-num_neighbors:]
        S_agg = np.sum(S_neighbors_filtered, axis=0)
        # bias for the original score 
        S_tilde = S_agg + bias*S_original*S_original
        # normalize scores 
        S_tilde = (S_tilde-np.mean(S_tilde))
        S = 10*sigmoid(sensitivity*S_tilde/np.max(S_tilde))
        '''
    
    # reconstruct xys_list (do not do in-place modification) 
    xys_filtered_list = []
    for i, s in enumerate(S):
        x,y = attraction_list[i].latlon
        xys_filtered_list.append((x,y,s))
    return xys_filtered_list

def get_dists(attraction_list):
    # get l2 distance between all node pairs
    # Note: this code can be optimized in future releases
    N = len(attraction_list)
    L = np.zeros(shape=(N,N)) 
    xy_list = [attraction.latlon for attraction in attraction_list]
    for i in range(N):
        for j in range(i, N):
            coord1 = np.array(xy_list[i])
            coord2 = np.array(xy_list[j])
            l = np.linalg.norm(coord1-coord2) 
            L[i][j] = l
            L[j][i] = l
    return L 

def sigmoid(x):
    # sigmoid function
    return 1 / (1 + np.exp(-x))

#####################################################################
# Synthetic Dataset Generation
#####################################################################
def gen_syn_loc(synthetic_params, restaurant_thresh, hotel_thresh):
    assert hotel_thresh + restaurant_thresh < 1
    xys_list = []
    for param in synthetic_params:
        # generate 2D gaussian
        num_samples, mu, sigma = param
        mu_x, mu_y = mu
        sigma_x, sigma_y = sigma
        for _ in range(num_samples):
            # sample data points 
            x = random.gauss(mu_x, sigma_x)
            y = random.gauss(mu_y, sigma_y)
            s = 10*random.random()
            xys_list.append((x,y,s))
    random.shuffle(xys_list)
    json_list = []
    for xys in xys_list:
        x,y,s = xys
        is_hotel = random.random() < hotel_thresh
        is_restaurant = random.random() < restaurant_thresh
        locale_type = 'hotel' if is_hotel else ('restaurant' if is_restaurant else 'other')
        metadata_google = json.dumps({'latitude': x, 'longitude':y, 'google_score': s, 'type': locale_type, 'metadata': random.random()*1000})
        json_list.append(metadata_google)
    print(json_list)
    return json_list

float_list = lambda x_list: [float(x) for x in x_list]
def json2attraction(metadata_google):
    metadata_google_decoded = json.loads(metadata_google)

    latlon_raw = (metadata_google_decoded['latitude'],metadata_google_decoded['longitude'])
    score_raw = metadata_google_decoded['google_score']
    is_restaurant_raw = metadata_google_decoded['type'] == 'restaurant'
    is_hotel_raw = metadata_google_decoded['type'] == 'hotel'

    latlon = tuple(float_list(latlon_raw))
    score = float(score_raw)
    is_restaurant = bool(is_restaurant_raw)
    is_hotel = bool(is_hotel_raw)
    assert len(latlon) == 2
    return Attraction(latlon, score, is_hotel, is_restaurant, metadata_google)

def attraction2xys(attraction):
    x,y = attraction.latlon
    s = attraction.score
    return x,y,s

#####################################################################
# Plotting Utilities 
#####################################################################
def plot(xys_list, ax, k=None):
    color = 'grey'
    color_sel = 'red'
    if k is None:
        # setting up plottig parameters 
        x_list = [xys[0] for xys in xys_list]
        y_list = [xys[1] for xys in xys_list]
        s_list = [xys[2] for xys in xys_list]
        plot_helper(x_list, y_list, s_list, ax, color)
    else:
        # setting up plottig parameters 
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
    # actual plotting code
    ax.scatter(x_list, y_list, c=color)
    for i, s in enumerate(s_list):
        ax.annotate('{0:.2f}'.format(s), (x_list[i], y_list[i]))

#####################################################################
# Main Function 
#####################################################################
def main():
    # dataset configurations
    synthetic_params = [(35, (0,-10), (5,6)), (33, (12,8), (7,7))]
    # clustering configurations
    num_days=2
    # topk?
    k = 8 
    # filtering configurations
    max_itr=5000
    num_neighbors=k
    sensitivity=8
    bias=10

    # generate synthetic datasets
    random.seed(123)
    json_list = gen_syn_loc(synthetic_params, 0.05, 0.1)
    
    attraction_list = [json2attraction(metadata_google) for metadata_google in json_list]
    xys_list = [attraction2xys(attraction) for attraction in attraction_list]

    # generate cluster by number of days
    cluster_id_list, centroids = get_cluster_id_list(attraction_list, num_days)
    attraction_list_clusters = cluster_attraction_list(cluster_id_list, attraction_list, num_days)

    # plotting the clusters
    xys_list_temp = xys_list.copy()
    xys_list_temp.extend([(centroid[0], centroid[1], float('inf')) for centroid in centroids])
    ax = plt.gca()
    ax.set_title('Input Scores and Cluster Centroids')
    ax.set_xlabel('X location')
    ax.set_ylabel('Y location')
    plot(xys_list_temp, ax, k=num_days)
    plt.show()

    for i, attraction_list in enumerate(attraction_list_clusters):
        # curate the score list of each cluster
        xys_filtered_list = filter_attraction_list(attraction_list, max_itr=max_itr, num_neighbors=num_neighbors, sensitivity=sensitivity, bias=bias)

        # Show the pre- and post-filtering scores and selections
        '''
        ax1 = plt.subplot(121)
        ax1.set_title('Input Scores (topk={}) for Cluster {}'.format(k, i))
        ax1.set_xlabel('X location')
        ax1.set_ylabel('Y location')
        plot(xys_list, ax1, k=k)
        '''
        ax2 = plt.subplot(111)
        ax2.set_title('Filtered Scores (topk={}) for Cluster {}'.format(k, i))
        ax2.set_xlabel('X location')
        ax2.set_ylabel('Y location')
        plot(xys_filtered_list, ax2, k=k)
        plt.show()

if __name__ == '__main__':
    main()
