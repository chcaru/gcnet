print 'Loading dependencies...'
import math, sys
import numpy as np

from keras import backend as K
from keras.layers import Dense, Dropout, Input, Flatten, LSTM, TimeDistributed, RepeatVector, Embedding, merge, Bidirectional, Lambda
from keras.models import Model


# Set your pretrained weights here!
PRETRAINED_WEIGHTS = ''


if len(PRETRAINED_WEIGHTS) is 0:
    print '\nYou must set PRETRAINED_WEIGHTS to your weights!\n'
    exit()

Bi = Bidirectional
_LSTM = LSTM
LSTM = lambda s, rs=True, gb=False, ur=True: _LSTM(s, return_sequences=rs, consume_less='gpu', unroll=ur, go_backwards=gb)
BLSTM = lambda s, rs=True, gb=False, ur=True: Bi(LSTM(s, rs, gb, ur))
Sum = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))

vocabSize = 6001
wordVectorSize = 300
captionLength = 16
gifFrames = 16

print 'Building GCNet...'
gifFramesVGG16 = Input(shape=(gifFrames, 1000))

# Trained Word Embeddings
embeddingMatrix = np.load('./embeddingMatrix.' + str(vocabSize - 1) + '.npy')
WordEmbedding = Embedding(input_dim=vocabSize, 
    output_dim=wordVectorSize, 
    mask_zero=True, 
    input_length=captionLength - 1, 
    weights=[embeddingMatrix], 
    trainable=False)

captionInput = Input(shape=(captionLength - 1,), dtype='int32')
wordVectorizedCaption = WordEmbedding(captionInput) 

dGIFFramesVGG16 = Dropout(.15)(gifFramesVGG16)
gifEncoder = BLSTM(1024)(dGIFFramesVGG16)
gifEncoder = Dropout(.15)(gifEncoder)
gifEncoder = LSTM(1024, rs=False)(gifEncoder)

gifFramesVGG16Sum = Sum(gifFramesVGG16)

encodedGIF = merge([gifEncoder, gifFramesVGG16Sum], mode='concat')

repeatedEncodedGIF = RepeatVector(captionLength - 1)(encodedGIF)

concatenatedWordVectorsAndEncodedGIF = merge([wordVectorizedCaption, repeatedEncodedGIF], mode='concat')

concatenatedWordVectorsAndEncodedGIF = Dropout(.15)(concatenatedWordVectorsAndEncodedGIF)
gifCaptionEncoder = BLSTM(1024)(concatenatedWordVectorsAndEncodedGIF)
gifCaptionEncoder = Dropout(.15)(gifCaptionEncoder)
gifCaptionEncoder = LSTM(1024, rs=False)(gifCaptionEncoder)

concatenatedEncoders = merge([gifCaptionEncoder, encodedGIF], mode='concat')

concatenatedEncoders = Dropout(.15)(concatenatedEncoders)
nextWord = Dense(vocabSize, activation='softmax')(concatenatedEncoders)

GCNet = Model([gifFramesVGG16, captionInput], nextWord)
GCNet.compile(loss='sparse_categorical_crossentropy', 
    optimizer='rmsprop', 
    metrics=['accuracy'])
GCNet.summary()

GCNet.load_weights(PRETRAINED_WEIGHTS)

print 'Loading vocab...'
vocab = open('./vocab.' + str(vocabSize - 1) + '.txt', 'r').readlines()
vocab = { (index + 1): word for word, index in [(line.split(' ')[0], i) for line, i in zip(vocab, xrange(len(vocab)))] }
vocab[0] = '_'

def generateCaption(oneHotCaption, oneHot=True):

    caption = ''

    for oneHotWord in oneHotCaption:

        index = np.argmax(oneHotWord) if oneHot else oneHotWord[0]
        word = vocab[index]
        caption += word + ' '

    return caption

# Within the last 625 GIFs used for validation only.
numValidation = 256
batchSize = 128

print 'Loading captions...'
dataY = np.load('dataY.captions.npy')
Y, dataYValidation = np.split(dataY, [-numValidation])
dataYIDsValidation, dataYWordsValidation = np.split(dataYValidation, [1], 1) 
dataYIDsValidation = dataYIDsValidation.reshape((len(dataYIDsValidation),)) 
dataYWordsValidation = dataYWordsValidation.reshape((len(dataYWordsValidation), dataYWordsValidation.shape[-1], 1))

print 'Loading precomputed VGG16 frames...'
precomputedVGG16Frames = np.load('./precomputedVGG16Frames.' + str(gifFrames) + '.npy')

print 'Starting test...'
print 'GIF ID : Generated Caption : Actual Caption'
print '-------------------------------------------'
lastStartIndex = 0
while True:

    batchIDs = dataYIDsValidation[lastStartIndex:lastStartIndex+batchSize]
    batchWords = dataYWordsValidation[lastStartIndex:lastStartIndex+batchSize]
    lastStartIndex += batchSize

    if len(batchIDs) <= 0:
        break

    batchImages = precomputedVGG16Frames[batchIDs]
    batchCaptions = np.zeros((len(batchIDs), captionLength - 1), dtype='int32')

    for i in range(captionLength):
        results = GCNet.predict([batchImages, batchCaptions])
        if i == captionLength - 1:
            batchCaptions = np.concatenate((batchCaptions, [[np.argmax(x)] for x in results]), 1)
        else:
            batchCaptions[:,i] = [np.argmax(x) for x in results]

    for i in range(len(batchCaptions)):
        print str(batchIDs[i]) + ' : ' + generateCaption(batchCaptions[i].reshape((captionLength, 1)), False) + ' : ' + generateCaption(batchWords[i], False)
