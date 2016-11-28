# Neural Network Generated GIF Captions | GCNet (GIF Caption Network)

## The goal of GCNet is to produce high quality GIF captions.

Below are GIFs from the [TGIF] dataset, and GCNet's generated captions for them. GCNet was not trained with these GIFs!

| GIF from [TGIF] Dataset | GCNet Generated GIF Caption |
|---|---|
| <img src="./imgs/TGIF/101108.gif" alt="TGIF" width="240">  | a monkey with an animal is eating something |
| <img src="./imgs/TGIF/101109.gif" alt="TGIF" width="240">  | a man is holding a microphone and moving his hands |
| <img src="./imgs/TGIF/101054.gif" alt="TGIF" width="240">  | a white car is driving down a road |
| <img src="./imgs/TGIF/101201.gif" alt="TGIF" width="240">  | a soccer player is scoring a goal in a football match |
| <img src="./imgs/TGIF/101123.gif" alt="TGIF" height="240"> | a dog is trying to catch a toy |
| <img src="./imgs/TGIF/101097.gif" alt="TGIF" height="240"> | a girl with blonde hair is talking and moving her head |
| <img src="./imgs/TGIF/101286.gif" alt="TGIF" height="240"> | a young woman in a car is driving and smiles |

# Architecture

## Input

1. GIF frames' precomputed VGG16 output (TODO: Create standalone GCNet that doesn't require precomputation)
2. In-progress GIF caption. This is a subcaption of the full caption (or the to be caption outside of training). See [Setup Step 7 - Data Expansion](#data-expansion). See [Obtaining a Caption with GCNet](#obtaining-a-caption-with-gcnet) for more details. 

## Output

1. Next word of the in-progress GIF caption. See Obtaining a Caption with GCNet for more details.


## Overview

GCNet can be thought of as computing: P(next word in caption | GIF, in-progress caption)

<img src="./imgs/gcnetOverview.png" alt="GCNet Architecture Overview" width="980">

## Obtaining a Caption with GCNet

GCNet generates a GIF's caption iteratively, requiring the GIF and its in-progress caption to be run through GCNet the number of times there are words in the caption. This is because GCNet computes the next word given an input GIF and in-progress caption. The first iteration's in-progress caption will consist of empty word indices (all zeros), producing the first word in the caption. This word becomes part of the in-progress caption, and is fed back into GCNet along with the same GIF, producing the second word, and so on... until the in-progress caption is at the caption's max length - 1, producing the last word in the caption. This results in the in-progress caption becoming the final generated caption for the given GIF. 

1. For right now, all input needs to be precomputed. Steps to do this are in the Setup section. 
2. For right now, once precomputed inputs are produced, use `gcnet.test.py` by changing the precomputed file references to your own.

(TODO: Create standalone `gcnet.py` that takes a GIF file name as input from the command line and prints out the caption for the GIF. It will also download pretrained GCNet weights.)

## Pretrained components

1. ImageNet Trained [VGG16]
2. Stanford's [GloVe] Word Vectors (840B tokens, 2.2M vocab, 300D)

# Setup

From start to finish, this will take at least 6 hours if you have a gigabit internet connection, fast processor, a lot of memory (at least 64GB), and a good GPU :)

## 0. Requirements

- [Keras]
- [Theano] / [TensorFlow] (as [Keras Backend])
- GPU (optional, but HIGHLY recommended)
- python 2.7 (recommended [Anaconda])
- [node.js] 6.9.1
- A dataset that follows the data format described at the bottom


## 1. Provide Dataset

1. `mkdir data`
2. Place a dataset with the Data Format requirements described at the bottom of the page in the Data Format section into `./data/gif-url-captions.tsv` 

OR, if you don't have your own dataset, you may choose to use this dataset ([TGIF], [Kaggle]), please review their license! Follow the below steps to proceed with this option.

1. `mkdir data`
2. `wget https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/tgif-v1.0.tsv -O ./data/gif-url-captions.tsv`

## 2. Download
This will download all GIFs in the above dataset. If using [TGIF] as your dataset, this is ~120GB. Make sure you have enough room!

1. `mkdir gifs` 
2. `node download.js` 

- If you encounter errors while running this, increase `DOWNLOAD_INTERVAL` in `download.js`
- `download.js` will also strip out captions into `./captions.txt` for further processing

## 3. Prepare GIFs
This will split, resize, and save the resulting GIF frames such that for every GIF `./gifs/X.gif` with `N` frames, it will create frame PNGs `./gifs/X/X_[0..N].png`.

This doubles the size of the data, to ~250GB. If you would like to in place remove processed GIFs (keep size at ~120GB), then you must set `removeProcessedGifs = True` in `prepareGifs.py`

1. `python -i prepareGifs.py`

## 4. Clean Captions
This will attempt to normalize the captions by removing unneeded punctuation and expressions, saving them to `./clean.captions.txt`

1. `python -i cleanCaptions.py`

## 5. Filter Captions
`filterCaptions.py` will compute the vocab (and save to `vocab.#vocabSize.txt`), compute the embedding matrix (and save to `embeddingMatrix.#vocabSize.npy`), filter out captions that are low quality (default < 90% of words in caption are in vocab), and finally compute vocab indexed captions (and save to `dataY.captions.#captionLength.npy`)

1. `wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O ./data/glove.840B.300d.zip`
2. `unzip ./data/glove.840B.300d.zip -d ./data/glove`
3. `python -i filterCaptions.py`

<img src="./imgs/filterCaptions.png" alt="GCNet Compute Vocab and Embedding Matrix" width="980">
<img src="./imgs/computeCaptionVectors.png" alt="GCNet Compute GIF Caption Vectors" width="980">

## 6. Precompute GIF frames' VGG16 output
Depending on your GPU, this step can take a while. On a GTX 1080, it takes about 3 hours using default settings (~1.65M images). Saves precomputed [VGG16] GIF frames to `precomputedVGG16Frames.#gifFrames.npy` (~6GB)

1. `python -i precomputeVGG16.py`

<img src="./imgs/prepareGifs.png" alt="GCNet Precompute GIF Frame's VGG16 Output" width="980">

## 7. Running GCNet
If you changed any of the variables in either step 5 or 6, then you will need to change the corresponding variables in `gcnet.py`

This will load all precomputed data, build GCNet, expand the data (see figure below), and start training.

1. `python -i gcnet.train.py`

### Data Expansion
<img src="./imgs/expandCaptions.png" alt="GCNet Expand GIF Captions" width="980">

## 8. Test Trained GCNet
If you changed any of the variables in either step 5 or 6, then you will need to change the corresponding variables in `gcnet.test.py`

1. Set `PRETRAINED_WEIGHTS` in `gcnet.test.py` to the location of your trained weights file
2. `python -i gcnet.test.py`
3. Enjoy!

# Data Format
Provide a list of GIF urls and corresponding captions with the following data format:

Each new line will contain:

`gif-url gif-caption` (such that `gif-url` and `gif-caption` are separated by a tab)

For example:

```
http://doggif.gif a dog playing catch
http://catgif.gif a cat walking around
```

Above are instructions for obtaining a dataset that meets this format. (GCNet does not require the above dataset as long as the aforementioned data format is followed)

   [GloVe]: <http://nlp.stanford.edu/projects/glove/>
   [VGG16]: <https://arxiv.org/abs/1409.1556>
   [GIF Caption Dataset]: <https://arxiv.org/abs/1604.02748>
   [TGIF]: <https://github.com/raingo/TGIF-Release>
   [Anaconda]: <https://www.continuum.io/downloads>
   [Keras]: <https://keras.io/>
   [Keras Backend]: <https://keras.io/backend/>
   [Theano]: <http://deeplearning.net/software/theano/>
   [TensorFlow]: <https://www.tensorflow.org/>
   [node.js]: <https://nodejs.org/en/>
   [Kaggle]: <https://www.kaggle.com/raingo/tumblr-gif-description-dataset>