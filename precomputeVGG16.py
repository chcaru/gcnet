print 'Loading dependencies...'

import os, math, sys
import numpy as np

from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b

from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg16 as vgg16
from keras.layers import Input


gifDir = './gifs/'
minNumFrame = 16
numFrameSamples = 16


def preprocessImage(imagePath):
    img = load_img(imagePath, target_size=(244, 244))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    img = img.reshape(img.shape[1:])
    return img

def getImage(imagePath):
    return preprocessImage(imagePath)

print 'Computing frames...'
numSkips = 0
gifFramePaths = [None] * 102068
gifIds = []

for i in range(102068):
    ii = str(i)
    if os.path.isdir(gifDir + ii):
        numFrames = len(os.listdir(gifDir + ii))
        if numFrames >= minNumFrame:
            frameNums = np.linspace(0, numFrames - 1, numFrameSamples, dtype='int32')
            framePaths = [gifDir + ii + '/' + ii + '_' + str(frameNum) + '.png' for frameNum in frameNums]
            gifFramePaths[i] = framePaths
            gifIds.append(i)
        else:
            numSkips += 1

gifIds = np.array(gifIds)
gifFramePaths = np.array(gifFramePaths)
print 'Skipped ' + str(numSkips) + ' gifs'

print 'Building VGG16 model...'
imageInput = Input(shape=(3, 244, 244))

# ImageNet Trained VGG16
vgg16Model = vgg16.VGG16(input_tensor=imageInput, weights='imagenet', include_top=True)

# This is the number of gifs
# The actual number of frames in a batch is (batchSize * numFrameSamples)
batchSize = 256
numBatches = len(gifIds) / batchSize + 1

vgg16Out = np.zeros((102068, numFrameSamples, 1000), dtype='float32')  

print 'Starting...'
i = 0
lastStartIndex = 0
while True:

    batchIDs = gifIds[lastStartIndex:lastStartIndex+batchSize]
    lastStartIndex += batchSize

    if len(batchIDs) <= 0:
        break

    batchImages = np.array([[getImage(framePath) for framePath in framePaths] for framePaths in gifFramePaths[batchIDs]], dtype='float32')
    batchImages = batchImages.reshape(((len(batchIDs) * numFrameSamples), 3, 244, 244))

    results = vgg16Model.predict(batchImages).reshape((len(batchIDs), numFrameSamples, 1000))

    vgg16Out[batchIDs] = results

    progress = int(math.floor(30.0 * (i + 1) / numBatches))
    progressBar = '\r\t' + str(i + 1) + '/' + str(numBatches) + '\t[' + ('=' * progress) + ('>' if 0 < progress < 30 else '') + ('.' * (30 - progress)) + ']'  

    sys.stdout.write(progressBar)
    sys.stdout.flush()
    i += 1

np.save('./precomputedVGG16Frames.' + str(numFrameSamples) + '.npy', vgg16Out)