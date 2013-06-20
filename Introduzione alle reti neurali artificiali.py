# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import random
import numpy as np
from PIL import Image

# <codecell>

def load_img(filename):
    img = Image.open(filename)
    return np.matrix([int(x != (255,255,255,0)) for x in img.getdata()]).transpose()

# <codecell>

def risultato_atteso(indice):
    return np.array([int(x == indice) for x in range(0,3)])
    #esempio: [0, 0, 1] se l'immagine è un 3

# <codecell>

esempi = [{
'immagine': load_img('lettere/%i%s.png' % (i, c)),
'categoria': risultato_atteso(i-1)
} for i in range(1,4) for c in 'abcd']

# <codecell>

# STAMPA DEGLI ESEMPI

def pprint(img):
	for x in range(8):
		for y in range(8):
			val = img[8*x + y]
			if val:
				print '#',
			else:
				print '.',
		print ''
	print ''

for i in range(len(esempi)):
    x = esempi[i]['immagine']
    e = esempi[i]['categoria']
    pprint(x)
    print e

# <codecell>

pesi = np.zeros((3, 64))

# <codecell>

epsilon = 0.2

runs = 0
errori = True

while errori:
    errori = False
    random.shuffle(esempi)
    for esempio in esempi:
        img, atteso = esempio['immagine'], esempio['categoria']
        prova = (pesi * img > 0).transpose()
        if not all(prova == atteso):
            errori = True
        
        delta = epsilon * img * (atteso - prova)
        pesi += delta.transpose()
        runs += 1
print 'Training completato in %i passi' % runs

# <codecell>

import pylab as pl

for i in range(0, 3):
    print 'Classe %i:' % (i + 1)
    x = np.reshape(pesi[i], (8,8))
    x = np.flipud(x) # in memoria le immagini sono caricate a testa in giù :)
    pl.pcolor(array(x))
    pl.colorbar()
    pl.show()

# <codecell>

for esempio in esempi:
    img, atteso = esempio['immagine'], esempio['categoria']
    prova = pesi *img > 0
    pprint(img)
    print 'riconosciuto come:', [i + 1 for i in range(len(prova)) if prova[i]]
    print '\n\n'

# <codecell>

# Dataset con immagini 16x16
data = file('semeion.data')
esempi16 = [(np.matrix(map(lambda x: float(x) > 0, l.split()[:256])).transpose(), np.array(map(int, l.split()[256:]))) for l in data]
random.shuffle(esempi16)

test_set = esempi16[:500]
esempi16 = esempi16[500:]

pesi16 = np.random.random((10, 256))

# <codecell>

epsilon = 0.2

runs = 0
errori = True

while errori:
    errori = False
    random.shuffle(esempi16)
    for esempio in esempi16:
        img, atteso = esempio
        prova = (pesi16 * img > 0).transpose()
        if not all(prova == atteso):
            errori = True
        
        delta = epsilon * img * (atteso - prova)
        pesi16 += delta.transpose()
    runs += 1
print 'Training completato in %i passi' % runs

# <codecell>

x = np.reshape(pesi16[0], (16,16))
x = np.flipud(x)
pl.pcolor(array(x))
pl.colorbar()
pl.show()

# <codecell>

def pprint16(img):
	for x in range(16):
		for y in range(16):
			val = img[16*x + y]
			if val:
				print '#',
			else:
				print '.',
		print ''
	print ''

print '% errori:',
x = sum([1 for img, atteso in esempi16 if not all((pesi16 * img > 0).transpose() == atteso)])/float(len(esempi16))
print x
if x:
    print "Prova a eseguire un altro ciclo di training!"

# <codecell>

print 'Test sugli esempi NON usati come training:\n\n\n' 

print '% errori:',
x = sum([1 for img, atteso in test_set if not all((pesi16 * img > 0).transpose() == atteso)])/float(len(test_set))
print x


print "alcuni esempi:"
for esempio in random.sample(test_set, 10):
    img, atteso = esempio
    prova = pesi16 * img > 0
    pprint16(img)
    print 'riconosciuto come:', [i for i in range(len(prova)) if prova[i]]
    print '\n\n'

# <codecell>


