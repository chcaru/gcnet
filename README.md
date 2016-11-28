# Neural Network Generated GIF Captions | GCNet (GIF Caption Network)

## The goal of this model is to produce high quality GIF captions.
This is accomplished by utilizing three components, two of which are pretrained and perform well in their orignal domain, and a third component that is described below.

## The two pretrained components are:
1. ImageNet Trained [VGG16]
2. Stanford's [GloVe] Word Vectors (840B tokens, 2.2M vocab, 300D)



# Setup

### Setup from start to finish will take at least 6 hours if you have a gigabit internet connection, fast processor, a lot of memory (at least 64GB), and a good GPU :)

## 0. Requirements
- [Keras]
- [Theano] / [TensorFlow] (as [Keras Backend])
- GPU (optional, but HIGHLY recommended)
- python 2.7 (recommended [Anaconda])
- [node.js] 6.9.1
- A dataset that follows the data format described at the bottom


## 1. Provide Dataset
1. `mkdir data`
2. Place a dataset with the above Data Format requirements in `./data/gif-url-captions.tsv` 

### OR, if you don't have your own dataset:
### You may choose to use this dataset ([TGIF], [Kaggle]), please review their license!
### This will download all GIFs in the above dataset. This is ~120GB. Make sure you have enough room!
1. `mkdir data`
2. `wget https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/tgif-v1.0.tsv -O ./data/gif-url-captions.tsv`

## 2. Download
1. `mkdir gifs` 
2. `node download.js` 

- If you encounter errors while running this, increase `DOWNLOAD_INTERVAL` in `download.js`
- `download.js` will also strip out captions into `./captions.txt` for further processing

## 3. Prepare GIFs
### This will split, resize, and save the resulting GIF frames such that for every GIF `./gifs/X.gif` with `N` frames, it will create frame PNGs `./gifs/X/X_[0..N].png`
### This doubles the size of the data, to about 250GB. If you would like to in place remove processed GIFs (keep size at ~120GB), then you must set `removeProcessedGifs = True` in `prepareGifs.py`
1. `python prepareGifs.py`

## 4. Clean Captions
This will attempt to normalize the captions by removing unneeded punctuation and expressions, saving them to `./clean.captions.txt`
1. `python cleanCaptions.py`

## 5. Filter Captions
### `filterCaptions.py` will compute the vocab (and save to `vocab.#vocabSize.txt`), compute the embedding matrix (and save to `embeddingMatrix.#vocabSize.npy`), filter out captions that are low quality (default < 90% of words in caption are in vocab), and finally compute vocab indexed captions (and save to `dataY.captions.#captionLength.npy`)
1. `wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O ./data/glove.840B.300d.zip`
2. `unzip ./data/glove.840B.300d.zip -d ./data/glove`
3. `python -i filterCaptions.py`

## 6. Precompute GIF frame's VGG16 output
### Depending on your GPU, this step can take a while. On a GTX 1080, it takes about 3 hours using default settings (~1.65M images). Saves precomputed [VGG16] GIF frames to `precomputedVGG16Frames.#gifFrames.npy` (~6GB)
1. `python -i precomputeVGG16.py`

## 7. Running GCNet
If you changed any of the variables in either step 5 or 6, then you will need to change the corresponding variables in `gcnet.py`
1. `python -i gcnet.py`

## 8. Test trained GCNet
If you changed any of the variables in either step 5 or 6, then you will need to change the corresponding variables in `gcnet.test.py`
1. Set `PRETRAINED_WEIGHTS` in `gcnet.test.py` to the location of your trained weights file
2. `python -i gcnet.test.py`
3. Enjoy!

# Data Format
### Provide a list of GIF urls and corresponding captions with the following data format:
Each new line will contain:

`gif-url gif-caption` (such that `gif-url` and `gif-caption` are separated by a tab)

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