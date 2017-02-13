import numpy as np
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

def compute_means(data):
    ret = np.empty((10,3072))
    for i in range(10):
        for j in range(data[i].shape[0]):
            ret[i] = np.add(ret[i], data[i][j])
        ret[i] = np.divide(ret[i], data[i].shape[0])
    return ret

    
sorted_data = read_files()
means = compute_means(sorted_data)
#
# input function
# 5 files-> pictures
#  
