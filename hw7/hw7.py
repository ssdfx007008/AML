from struct import unpack
import gzip
from numpy import zeros, uint8, float32
import numpy as np
from pylab import imshow, show, cm
import random

def get_labeled_data(imagefile, labelfile):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    # Open the images with gzip in read binary mode
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = zeros((N, rows, cols), dtype=float32)  # Initialize numpy array
    y = zeros((N, 1), dtype=uint8)  # Initialize numpy array
    for i in range(N):
        if i % 1000 == 0:
            print("i: %i" % i)
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = (-1)*(2.0*(random.random()<0.02)-1)*((tmp_pixel>0.5)*2-1)
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (x, y)


def view_image(image, label=""):
    """View a single image."""
    print("Label: %s" % label)
    imshow(image, cmap=cm.gray)
    show()
#%%


data=get_labeled_data("t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz")
output=np.copy(data)
#%%
miu=zeros((28,28))+1
miu_pre=zeros((28,28))+0
for imgIdx in [0]:
    while np.sum(abs(miu_pre-miu))>0.0000000000000001:
        miu_pre=miu
        for i in range(28):
            for j in range(28):
                neighbor=0
                if(i-1>0):
                    neighbor+=0.2*(2*data[0][imgIdx][i-1][j]-1)
                if(i+1<28):
                    neighbor+=0.2*(data[0][imgIdx][i+1][j]*2-1)
                if(j-1>0):
                    neighbor+=0.2*(data[0][imgIdx][i][j-1]*2-1)
                if(j+1<28):
                    neighbor+=0.2*(data[0][imgIdx][i][j-1]*2-1)
                        
                    xinput=2*data[0][imgIdx][i][j]
                    miu[i][j]= np.exp(neighbor+xinput)/(np.exp(neighbor+xinput)+np.exp(-neighbor-xinput))
    for i in range(28):
        for j in range(28):
            x=data[0][imgIdx][i][j]
            output[0][imgIdx][i][j]=(-1)**(miu[i][j]**((1+x)/2)*(1-miu[i][j])**((1-x)/2) <0.5)*x            
            print(output[0][imgIdx][i][j])
view_image(data[0][0],data[1][0])
view_image(output[0][0],output[1][0])

#print(data[0][5])
#%%





#%%



#%%
print(len(data))
#%%
view_image(data[0][3],data[1][1])
print(data[0][1][5][14])





