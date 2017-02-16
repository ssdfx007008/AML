import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA
import _pickle as cPickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo,  encoding='latin1')
    fo.close()
    return dict

def read_files():
##    ret = np.empty( (10,0,3072))
    ret = [[] for x in range(10)]

    for i in range(1,6):#files 1 to 5
        print("reading file " + str(i) + "...")
        unsorted = unpickle("cifar-10-batches-py/data_batch_" + str(i))
        for i in range(10000):
            type = unsorted['labels'][i]
##            ret[type] = np.append(ret[type], unsorted['data'][i], axis = 0)
            ret[type].append(unsorted['data'][i])
    return np.array(ret)

def read_meta():
    return unpickle("cifar-10-batches-py/batches.meta")['label_names']
    

def compute_means(data):
    ret = np.empty((10,3072))
    for i in range(10):
        for j in range(data[i].shape[0]):
            ret[i] = np.add(ret[i], data[i][j])
        ret[i] = np.divide(ret[i], data[i].shape[0])
    return ret

def make_dist_matrix(inputvectors, distfunction):
    ret = np.zeros( (inputvectors.shape[0],inputvectors.shape[0]) )
    for i in range(inputvectors.shape[0]):
        for j in range(inputvectors.shape[0]):
            ret[i][j] = distfunction(inputvectors[i], inputvectors[j])
    return ret

def euclidean_dist(a, b):
    return distance.euclidean(a,b)

def partB(means, labels):
    print('Begin part B')
    dist_matrix = make_dist_matrix(means, euclidean_dist)
    mds = manifold.MDS(n_components = 2, verbose = 1,max_iter=3000, n_jobs = 1, dissimilarity = 'precomputed')
    #do NOT set n_jobs to anything > 1 - it makes fit hang for some reason
    print('Fitting using MDS...')
    mds_out = mds.fit(dist_matrix).embedding_
    #print('Scaling')
    #mds_out *= np.sqrt((means ** 2).sum()) / np.sqrt((mds_out ** 2).sum())
    clf = PCA(n_components=2)
    print('Transforming')
    mds_out = clf.fit_transform(mds_out)
    print('Plotting')
    plt.scatter(mds_out[:,0], mds_out[:,1])
    
    for point in zip(mds_out[:,0], mds_out[:,1], range(0,10)):
        plt.annotate(labels[point[2]], xy = point[0:2], xytext = (6, -6), textcoords = 'offset pixels')
        
    plt.show()

    
sorted_data = read_files()
label_names = read_meta()
means = compute_means(sorted_data)
partB(means, label_names)
#
# input function
# 5 files-> pictures
#  
