
import gc, math, sys, re, os
import numpy as np
from sets import Set
from collections import Counter

# The max resulting size of the vocabulary.
# These are the top N most frequent words 
maxVocab = 6000

# % of words in a caption that must be int
# the vocab before it is dropped (below quality threshold)
captionQualityThreshold = 0.9

# The fewest number of words in a caption before it is dropped
minWords = 3

# Determines if the captions will be trimmed
trimMax = True

# if trimMax:
#   Captions over this length will be trimmed to it
# else:
#   The greatest number of words in a caption before it is dropped
maxTrimWords = 16

# The greatest number of words in a caption before it is dropped
# The idea behind this is that if it's too long then the trimmed
# caption won't have enough information to learn from
maxWords = 22

# Filters out words from captions that are not in the vocabulary
# (this can result in bad grammar / english)
# Generally this is OK if captionQualityThreshold is high enough
removeWordsOutsideVocabFromCaption = True

gifDir = './gifs/'
minNumFrame = 16

# This should be the cleaned text...
# Cleaned text should have a higher quality and reduced complexity
text = open('clean.captions.txt', 'r').read();

# A path to trained word embeddings
preTrainedWordEmbeddingsPath = './data/glove/glove.840B.300d.txt'
# The size of the word embedding vectors
preTrainedWordEmbeddingSize = 300

showDroppedItems = False

lines = text.split('\n')
lines.pop()
captionIds = [line.split('\t')[0] for line in lines]
captions = [line.split('\t')[1] for line in lines]
tokens = ' '.join(captions).split(' ')

uniqueTokens = Counter(tokens)
tokenLookup = { token: index for token, index in zip(sorted(uniqueTokens, key=uniqueTokens.get, reverse=True), xrange(len(uniqueTokens))) }
reverseTokenLookup = { value: key for key, value in tokenLookup.items() }
# reverseTokenLookup[0] = '*'

maxVocab = min(maxVocab, len(reverseTokenLookup))

# This can take up a lot of memory (~40GB) when using glove.840B.300d
# If you have less than 64GB of RAM, then use glove.6B.300d (or smaller)
preTrainedEmbeddingLookup = { word: vector for word, vector in [(l[0], np.asarray(l[1:], dtype='float32')) for l in [ l.split(' ') for l in open(preTrainedWordEmbeddingsPath, 'r').readlines() ]] }

embeddingMatrix = np.zeros((maxVocab + 1, preTrainedWordEmbeddingSize), dtype='float32')

vocabFile = open('vocab.' + str(maxVocab) + '.txt', 'w')
for i in range(maxVocab):
    vocabFile.write(reverseTokenLookup[i] + ' ' + str(uniqueTokens[reverseTokenLookup[i]]) + '\n')

    word = reverseTokenLookup[i] if reverseTokenLookup[i] != 'N' else 'number'

    wordVector = preTrainedEmbeddingLookup.get(word)

    if wordVector is not None:
        embeddingMatrix[i+1] = wordVector
    else:
        print reverseTokenLookup[i] + ' was not found in pre trained word embeddings'

vocabFile.close()
np.save('./embeddingMatrix.' + str(maxVocab) + '.npy', embeddingMatrix)

def wordToIndex(w):
    i = tokenLookup[w]
    return i + 1 if i + 1 < maxVocab else 0

def indexToWord(i):
    return reverseTokenLookup[i-1] if i > 0 else '*'

def quality(indices):
    return reduce((lambda x,y: x+y), map(lambda i: 1.0 if i > 0 else 0.0, indices)) / len(indices)

filtedCaptionsFile = open('filtered.captions.' + str(maxVocab) + '.txt', 'w')
encodedCaptions = np.zeros((len(captions), 1 + maxTrimWords), dtype='int32')
numKept = 0
tQuality = 0.0
ttQuality = 0.0
tLength = 0.0
ttLength = 0.0
for i in range(len(captions)):

    line = captions[i]

    words = line.split(' ')

    indices = map(wordToIndex, words)
    q = quality(indices)

    ttQuality += q
    ttLength += len(words)

    if not os.path.isdir(gifDir + str(i)):
        if showDroppedItems: print 'Dropping (no gif): ' + str(i) + ' ' + line
        continue;
    
    if len(os.listdir(gifDir + str(i))) < minNumFrame:
        if showDroppedItems: print 'Dropping (too few frames): ' + str(i) + ' ' + line
        continue;

    if len(words) < minWords:
        if showDroppedItems: print 'Dropping (too small): ' + str(i) + ' ' + line
        continue

    if len(words) > maxTrimWords:

        if not trimMax or len(words) > maxWords:
            if showDroppedItems: print 'Dropping (too big): ' + str(i) + ' ' + line
            continue
        else:
            words = words[:maxTrimWords]

    indices = map(wordToIndex, words)
    q = quality(indices)

    if q < captionQualityThreshold:
        if showDroppedItems: print 'Dropping (low quality): ' + str(i) + ' ' + line
        continue

    if removeWordsOutsideVocabFromCaption:
        indices = filter(lambda x: x > 0, indices)

    tQuality += q
    tLength += len(indices)

    encodedCaptions[numKept][0] = int(captionIds[i])
    encodedCaptions[numKept][1:len(indices)+1] = indices

    filtedCaptionsFile.write(str(captionIds[i]) + ' ' + (' '.join(map(indexToWord, indices))) + '\n')

    numKept += 1

filtedCaptionsFile.close()

encodedCaptions = encodedCaptions[:numKept]
np.save('dataY.captions.' + str(maxTrimWords) + '.npy', encodedCaptions)

tQuality /= numKept
tLength /= numKept

ttQuality /= len(captions)
ttLength /= len(captions)

print 'Captions kept: ' + str(numKept) + ' / ' + str(len(captions)) 
print 'Average quality: ' + str(tQuality) + ' / ' + str(ttQuality)  
print 'Average length: ' + str(tLength) + ' / ' + str(ttLength)  