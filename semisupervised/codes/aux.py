import numpy as np
import pandas as pd
import argparse
from icecream import ic

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='net')
args = parser.parse_args()

def convert_clustering_to_net(clustering_result):
    net = []
    for i in range(clustering_result.shape[0]):
        for item in clustering_result[i,]:
            net.append([i, item, 1])
    net = np.array(net)
    return(net)

# def convert_net_to_clustering(net):
#     for 

def build_network(characteristic_file_path, k, auto_k=False, method='knn'):
    patient_characteristic = pd.read_csv(characteristic_file_path, header=0)
    clustering_data = np.array(patient_characteristic.iloc[:,1:26])
    # ic()
    # ic(clustering_data)

    from sklearn.metrics import silhouette_score
    if method == 'knn':
        from sklearn.neighbors import NearestNeighbors
        print('knn is used for network building.')

        if auto_k == True:
            print('Optimal k is searched by metric.')
        
            clustering_scores = []
            for i in range(1,10):
                neighbors = NearestNeighbors(n_neighbors=i, algorithm='ball_tree').fit(clustering_data)
                distances, indices = neighbors.kneighbors(clustering_data)
                clustering_scores[k-1] = silhouette_score(clustering_data, indices)
            optimal_k = clustering_scores.index(max(clustering_scores))+1
            print('The best performing k is', optimal_k)

            neighbors = NearestNeighbors(n_neighbors=optimal_k, algorithm='ball_tree').fit(clustering_data)
            distances, indices = neighbors.kneighbors(clustering_data)
            net = neighbors.kneighbors_graph()
        else:
            print('k =', k, 'is used.')
            neighbors = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(clustering_data)
            distances, indices = neighbors.kneighbors(clustering_data)
            net = neighbors.kneighbors_graph()
    # net = convert_clustering_to_net(indices)
    return(net)

def main():
    if args.output == 'net':
        net = build_network('../data/covid/COVID_encoded.csv', k=5)
        print('Saving network...')
        np.savetxt('../data/covid/net.txt', net, fmt='%i', delimiter='\t')
        # ic(net)
        # ic(clustering_result[1,])
        # print(clustering_result)

if __name__ == '__main__':
    main()