
import os
from PIL import Image

# VGG16 input size...
compressionSize = 224, 224

# Delete a gif once it's been split into frames
removeProcessedGifs = False

def iter_frames(im):
    try:
        i= 0
        while 1:
            im.seek(i)
            imframe = im.copy()
            if i == 0: 
                palette = imframe.getpalette()
            else:
                imframe.putpalette(palette)
            yield imframe
            i += 1
    except EOFError:
        pass

def gen_frames(im, name):
    for i, frame in enumerate(iter_frames(im)):
        x = (name + '_%d.png') % i
        frame.thumbnail(compressionSize, Image.ANTIALIAS)
        frame.save(x, optimize=True, quality=100, **frame.info)
        frame.close()

for i in range(0, 102068):
    ii = str(i)
    if os.path.isfile('./gifs/' + ii + '.gif'):
        print i
        try:
            os.mkdir('./gifs/' + ii)
            im = Image.open('./gifs/' + ii + '.gif')
            gen_frames(im, './gifs/' + ii + '/' + ii)
        except Exception:
            continue
        im.close()
        if removeProcessedGifs:
            os.remove('./gifs/' + ii + '.gif')
