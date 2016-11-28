
import re

fileName = './captions.txt'

lines = open(fileName, 'r').read().lower().split('\n')

text = '\n'.join([line.split('\t')[1] for line in lines])

r = re.sub('(\([^()]*\))', '', text)
r = re.sub('&amp;(amp;)*', 'and', text)

remove = ['amp;', '/', '[', ']', '&', '*', '_', ':', ';', '(', ')', '{', '}', '|', '\\', '~', '%', '+', '>', '<', '@', '$', '=', '#', '^', '`', ',', '.', '!', '?', '"']
for removal in remove:
    r = r.replace(removal, ' ')

r = r.replace('-', ' ').replace("'", '')
r = r.replace('     ', ' ').replace('    ', ' ').replace('   ', ' ').replace('  ', ' ')

r = r.split('\n')

outFile = open('./clean.captions.txt', 'w')
for cleanedLine, originalLine in zip(r, lines):
    outFile.write(originalLine.split('\t')[0] + '\t' + cleanedLine.strip(' ') + '\n')
outFile.close()