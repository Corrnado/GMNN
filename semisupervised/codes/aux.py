import numpy as np
import pandas as pd
import argparse
from icecream import ic

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='net')
args = parser.parse_args()

def convert_clustering_to_net(clustering_result, method):
    net = []
    if method == 'knn':
        for i in range(clustering_result.shape[0]):
            for item in clustering_result[i,]:
                net.append([i, item, 1])
        net = np.array(net)
    elif method == 'kmeans':
        # adjacency_matrix = np.zeros((len(clustering_result), len(clustering_result))))
        for i in range(len(clustering_result)):
            current_group = clustering_result[i]
            for j in range(i,len(clustering_result)):
                if clustering_result[j] == current_group:
                    net.append([i,j,1])
    elif method == 'distance':
        for i in range(clustering_result.shape[0]):
            for j in range(i,clustering_result.shape[0]):
                if clustering_result[i,j] < 0.2:
                    net.append([i,j,1])
    return(net)

# def convert_net_to_clustering(net):
#     for 

def build_network(characteristic_file_path, k, auto_k=False, filter=True, method='knn'):
    patient_characteristic = pd.read_csv(characteristic_file_path, header=0)
    if filter == True:
        patient_list = pd.read_csv('../data/covid/sbu_patient_list.csv',header=None)
        patient_list = patient_list.iloc[:,0].tolist()
        # ic(patient_list)
        patient_characteristic = patient_characteristic[patient_characteristic['to_patient_id'].isin(patient_list)]
        # ic(patient_characteristic)
        patient_characteristic = patient_characteristic.sort_values(by=['to_patient_id'])
    clustering_data = np.array(patient_characteristic.iloc[:,1:58])
    # ic(patient_characteristic)
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
            # net = neighbors.kneighbors_graph()
            net = convert_clustering_to_net(distances, method=method)
    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        print('kmeans is used for network building.')

        kmeans = KMeans(n_clusters=k, random_state=0).fit(clustering_data)
        # print(kmeans.labels_.tolist())
        cluster_assignment = kmeans.labels_
        # print(cluster_assignment)
        net = convert_clustering_to_net(cluster_assignment, method=method)
    elif method == 'distance':
        print('Euclidean distance is used for network building.')
        distances = np.zeros((clustering_data.shape[0],clustering_data.shape[0]))
        for i in range(clustering_data.shape[0]):
            for j in range(i,clustering_data.shape[0]):
                distance = np.linalg.norm(clustering_data[i,:]-clustering_data[j,:])
                distances[i,j] = distance
                distances[j,i] = distance
        distances = (distances - distances.min())/(distances.max() - distances.min())
        net = convert_clustering_to_net(distances, method=method)
    return(net)

def build_feature(patient_feature_file_path, normalize=True):
    patient_feature = pd.read_csv(patient_feature_file_path, header=None, index_col=0)
    # patient_feature = patient_feature.sort_values(by=[0])
    patient_feature = patient_feature.sort_index()
    if normalize:
        patient_feature = (patient_feature - patient_feature.min().min())/(patient_feature.max().max() - patient_feature.min().min())
    patient_feature = patient_feature.to_numpy()
    for i in range(patient_feature.shape[0]):
        feature_file = open('../data/covid/feature.txt', 'a')
        feature_file.write(str(i)+'\t')
        current_non_zero_index = np.where(patient_feature[i,] != 0)[0]
        # ic(current_non_zero_index)
        for item in current_non_zero_index:
            feature_file.write(str(item)+':'+str(patient_feature[i,item])+'\t')
        feature_file.write('\n')
        feature_file.close()

def encode_label(characteristic_file_path, filter=True):
    patient_characteristic = pd.read_csv(characteristic_file_path, header=0)
    if filter == True:
        patient_list = pd.read_csv('../data/covid/patient_list.csv',header=None)
        patient_list = patient_list.iloc[:,0].tolist()
        patient_characteristic = patient_characteristic[patient_characteristic['PATIENT_ID'].isin(patient_list)]
        patient_characteristic = patient_characteristic.sort_values(by=['PATIENT_ID'])
    patient_label = patient_characteristic.iloc[:,27]
    encoded_label = np.zeros(len(patient_label))
    encoded_label[np.where(patient_label == 'Y')[0]] = 1
    return(encoded_label)

def create_misc(test=30, train=30, dev=30, total=102):
    if (test+train+dev) > total:
        print('Invalid inputs!')
        exit()
    train_index = range(train)
    train_index = np.array(train_index)
    np.savetxt('../data/covid/train.txt', train_index, fmt='%i')

    dev_index = range(train, train+dev)
    dev_index = np.array(dev_index)
    np.savetxt('../data/covid/dev.txt', dev_index, fmt='%i')

    test_index = range(total-test, total)
    test_index = np.array(test_index)
    np.savetxt('../data/covid/test.txt', test_index, fmt='%i')


def main():
    if args.output == 'net':
        net = build_network('../data/covid/sbu_token_normalized.csv', k=20, method='distance')
        # print(net)
        print('Saving network...')
        np.savetxt('../data/covid/sbu_net.txt', net, fmt='%i', delimiter='\t')
        # ic(net)
        # ic(clustering_result[1,])
        # print(clustering_result)
    elif args.output == 'feature':
        build_feature('../data/covid/SBU_features_1024.csv')
    elif args.output == 'label':
        labels = encode_label('../data/covid/COVID_encoded.csv')
        np.savetxt('../data/covid/label.txt', np.dstack((np.arange(0, labels.size), labels))[0], fmt='%i', delimiter='\t')
    elif args.output == 'misc':
        create_misc(test=30, train=30, dev=30, total=102)
    else:
        print('Invalid input!')


if __name__ == '__main__':
    main()