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
#    print(np.isnan(np.min(ret)))
    return ret

    
def make_dist_matrix(inputvectors):
    ret = np.zeros( (inputvectors.shape[0],inputvectors.shape[0]) )

    for i in range(inputvectors.shape[0]):
        for j in range(inputvectors.shape[0]):
#            print(inputvectors[i], inputvectors[j])
            ret[i][j] = euclidean_dist(inputvectors[i], inputvectors[j])
    return ret

def euclidean_dist(a, b):
    return np.linalg.norm(a-b)

    
def make_new_dist_matrix(inputvectors,mean_matrix):
    ret = np.zeros( (inputvectors.shape[0],inputvectors.shape[0]) )

    for i in range(inputvectors.shape[0]):
        for j in range(inputvectors.shape[0]):
            ret[i][j] = 1/2*(new_dist(inputvectors[i], inputvectors[j],mean_matrix[i],mean_matrix[j])+new_dist(inputvectors[j], inputvectors[i],mean_matrix[j],mean_matrix[i]))
    return ret

def new_dist(a, b,mean_a,mean_b):
    pca=PCA(n_components=20)
    transformed=pca.fit_transform(b)
    inverse=pca.inverse_transform(transformed)
    for line in inverse:
        line=line+mean_a-mean_b
#        s = np.sqrt(np.sum((data[i]-mean_matrix[i]-inverse)**2))
    ret = np.sqrt(np.sum((a-inverse)**2)/5000)
    print (ret)
    return ret
    
def partA(data,mean_matrix,labels):
    ret = np.zeros( 10 )

    for i in range(10):
        #pca on data[i]
#        print(data[i].shape)
#        mds = manifold.MDS(n_components = 20, verbose = 1,max_iter=3000, n_jobs = 1, dissimilarity = 'precomputed')
#        ret[i] = np.add(mds.fit_transform(data[i]))
        pca=PCA(n_components=20)
        transformed=pca.fit_transform(data[i])
        inverse=pca.inverse_transform(transformed)
#        s = np.sqrt(np.sum((data[i]-mean_matrix[i]-inverse)**2))
        ret[i] = np.sqrt(np.sum(((data[i]-inverse))**2)/5000)
        print (ret[i])
#        ret[i]=np.add(ret[i],pca.explained_variance_ratio_)
#    print(np.isnan(np.min(ret)))
    n_groups = 10
    
    fig, ax = plt.subplots()
    
    index = np.arange(n_groups)
    bar_width = 0.35
    
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    
    rects1 = plt.bar(index, ret, bar_width,
                     alpha=opacity,
                     color='b',
                     error_kw=error_config,
                     label='Error For Part A')
    
    plt.xlabel('Categories')
    plt.ylabel('Scores')
    plt.title('Errors by Category')
    plt.xticks(index + bar_width / 2, labels)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    return 0
    
def partB(mean_matrix, labels):
    print('Begin part B')
    dist_matrix = make_dist_matrix(mean_matrix)
    if(np.isnan(np.min(mean_matrix)) or np.isinf(np.min(mean_matrix))):
        print('mean matrix is erroneous')
    if(np.isnan(np.min(dist_matrix)) or np.isinf(np.min(dist_matrix))):
        print('dist_matrix is erroneous')
#    print (dist_mat6rix)
    pca = PCA(n_components = 2)
    #do NOT set n_jobs to anything > 1 - it makes fit hang for some reason
    print('Fitting using MDS...')
    pca_out = pca.fit_transform(dist_matrix)
    print('Scaling')
#    pca_out *= np.sqrt((mean_matrix ** 2).sum()) / np.sqrt((pca_out ** 2).sum())
    if(np.isnan(np.min(pca_out)) or np.isinf(np.min(pca_out))):
        print('mds_out is erroneous')
    print(pca_out.shape)
#    pca = PCA(n_components=2)
#    print('Transforming')
#    mds_out = pca.fit_transform(mds_out)
    print('Plotting')
    plt.figure(1)
    plt.subplot(111)
    plt.scatter(pca_out[:,0], pca_out[:,1])
    
    for point in zip(pca_out[:,0], pca_out[:,1], range(0,10)):
        plt.annotate(labels[point[2]], xy = point[0:2], xytext = (6, -6), textcoords = 'offset pixels')
        
    plt.show(block=False)
    return plt.subplot(111)


def partC(data,mean_matrix, labels):
    print('Begin part B')
    dist_matrix = make_new_dist_matrix(data,mean_matrix)
    if(np.isnan(np.min(mean_matrix)) or np.isinf(np.min(mean_matrix))):
        print('mean matrix is erroneous')
    if(np.isnan(np.min(dist_matrix)) or np.isinf(np.min(dist_matrix))):
        print('dist_matrix is erroneous')
#    print (dist_mat6rix)
    pca = PCA(n_components = 2)
    #do NOT set n_jobs to anything > 1 - it makes fit hang for some reason
    print('Fitting using MDS...')
    pca_out = pca.fit_transform(dist_matrix)
    print('Scaling')
    if(np.isnan(np.min(pca_out)) or np.isinf(np.min(pca_out))):
        print('mds_out is erroneous')
    print(pca_out.shape)

    print('Plotting')
    plt.figure(1)
    plt.subplot(111)
    plt.scatter(pca_out[:,0], pca_out[:,1])
    
    for point in zip(pca_out[:,0], pca_out[:,1], range(0,10)):
        plt.annotate(labels[point[2]], xy = point[0:2], xytext = (6, -6), textcoords = 'offset pixels')
        
    plt.show(block=False)
    return plt.subplot(111)


    
sorted_data = read_files()
label_names = read_meta()
means = compute_means(sorted_data)
#pcas = compute_pcas(sorted_data)
#figure_a = partA(sorted_data,means,label_names)
#figure_b = partB(means, label_names)
figure_c = partC(sorted_data,means, label_names)

print('B completed')
#
# input function
# 5 files-> pictures
#  
