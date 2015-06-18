import numpy as np
from sklearn.linear_model import Perceptron
from itertools import izip
from multiprocessing import Pool
from operator import itemgetter
from random import randint

templateFts = ['t[l-1]','w[l]','t[l]','t[l+1]','"_".join(t[l+1:r])','t[r-1]','w[r]','t[r]','t[r+1]']
templates = [[1,2,6,7],
             [1,2,6],
             [1,6,7],
             [1,2,7],
             [2,6,7],
             [1,6],
             [2,7],
             [1,2],
             [6,7],
             [1],[2],[6],[7],
             [2,4,7],
             [2,3,5,7],
             [2,3,7],
             [2,3,7,8],
             [2,7,8],
             [0,2,7,8],
             [0,2,7],
             [0,2,5,7],
             [2,5,7]]

def main():
	#'''
	# Prepare training data
	global sentences
	global tags
	sentences, tags, sourceOrder = readTaggedData('Data/europarl-v7-h102000-tok-leq10.de-en.tagged.de')
	alignments = readAlignments('Data/europarl-v7-h102000-10.de-en.gdfa')
	global targetOrder
	targetOrder = getTargetOrder(sentences, alignments)
	
	global features
	features = initFeatures()
	print len(features), 'features found'
	
	#reorderingTraining(sentences, tags, sourceOrder, targetOrder)
	#'''

	print getFeatureValue(['t[l-1]','w[l]','"_".join(t[l+1:r])'], ['I', 'dont', 'care', 'yo'], ['A', 'B', 'C', 'D'], 0, 3)

def readTaggedData(sFile):
	sentences = []
	tags = []
	sourceOrder = []
	with open(sFile, 'rU') as sens:
		for line in sens:
			words = np.array([w.split('_') for w in line.strip().split(' ')])
			sentences.append(words[:,0])
			tags.append(words[:,1])
			sourceOrder = range(len(words))
			break
			#if len(sentences) == 10000: break
	return sentences, tags, sourceOrder 

def readAlignments(aFile):
	alignments = []
	with open(aFile, 'rU') as aligns:
		for line in aligns:
			align = line.strip().split(' ')
			alignments.append([map(int, a.split('-')) for a in align])
			#if len(alignments) == 10000: break
			break
	return alignments

def getTargetOrder(sentences, alignments):
	targetOrder = []
	for s, aligns in enumerate(alignments):
		# keep alignment to leftmost target word
		aligns = np.array(dict(reversed(aligns)).items())
		# align unaligned source words to the null word
		unaligned = list(set(range(len(sentences[s]))) - set(aligns[:,0]))
		if unaligned:
			# N.B. alignment indexing starts at 0
			aligns = np.concatenate([[[i, -1] for i in unaligned], aligns])
		# sort by target order
		tOrder = np.array(sorted(aligns, key=lambda a: a[1]))
		# retrieve source word indexes in target order
		targetOrder.append(tOrder[:,0])
	return targetOrder

def reorder(source, order):
	return [source[i] for i in order]
    
def genPermutations(sen, n):
	permutations = []
	while len(permutations) < n:
		perm = sorted([randint(0,len(sen)) for r in range(3)])
		if perm[0] != perm[2]:
			permutations.append(perm)
	return perm

def permute(sen, i, j, k):
	return sen[:i] + sen[j+1:k+1] + sen[i:j+1] + sen[k+1:]

def getFeatureVectors(sourceOrder):
	vectors = []
	for sen, tag, sOrder, tOrder in izip(sentences, tags, sourceOrder, targetOrder):
		sample = []
		w = reorder(sen, order)
		t = reorder(tag, order)
		for l in xrange(len(sen)):
			for r in xrange(1, len(sen)):
				entry = []
				for tem in templates:
					tem = [templateFts[i] for i in tem]
					value = getFeatureValue(tem, w, t, l, r)
					if value in features:
						entry.append(features.index(value))
						extended = getExtendedValue(value, l, r)
						if extended in features:
							entry.append(features.index(extended))
				if tOrder.index(sOrder[l]) < tOrder.index(sOrder[r]):
					entry.append(0) # don't invert
				else:
					entry.append(1) # invert
				sample.append(entry)
		vector.append(sample)
	return vectors

def getFeatureValue(template, w, t, l, r):
	value = ''
	for part in template:
		if 'l-1' in part and not l>0:
			continue
		if 'r+1' in part and not r<len(w)-2:
			continue
		if 'b' in part and not r-l<5:
			continue
		result = eval(part)
		value += '_'+result
	return value[1:]

def getExtendedValue(featureValue, l, r):
	d = r-l
	if d > 10:
		d = '>10'
	elif d > 5:
		d = '>5'
	return featureValue+'_'+str(d)

def initFeatures():
	features = []
	s = 0
	for sen, tag in izip(sentences, tags):
		s += 1
		if s % 10 == 0: print s
		for l in xrange(len(sen)):
			for r in xrange(l+1, len(sen)):
				for tem in templates:
					tem = [templateFts[i] for i in tem]
					value = getFeatureValue(t, sen, tag, l, r)
					features.append(value)
					features.append(getExtendedValue(value, l, r))
	return list(set(features))

def localSearch(sens, source, nperms):
	neighbors = []
	for sen, sOrder, perms in izip(sens, source, nperms):
		beta = defaultdict(0)
		delta = defaultdict(0)
		for i,j,k in perms:
			delta[i, k] = -float('inf')
				for j in xrange(i+1, k-1):
					delta[i, j, k] = getBenefit(sen, i, j, k)
					beta[i, k] = max(delta[i, k], )
	return beta[0, n]

def getPreference(sen, l, r):
	pass

def getBenefit(sen, i, j, k):
	if i is j or j is k:
		return 0
	delta = getBenefit(sen, i, j, k-1)
	delta += getBenefit(sen, i+1, j, k)
	delta -= getBenefit(sen, i+1, j, k-1)
	delta += getPreference(sen, k, i+1)
	delta -= getPreference(sen, i+1, k)
	return delta 

def getTrainingVectors():
	features = calculateFeatures()

def reorderingTraining(sourceOrder, iters=10):
	#perceptron = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, n_iter=1, shuffle=True, verbose=0, eta0=1.0,
    #                        n_jobs=1, random_state=0, class_weight=None, warm_start=False)
	
	for i in xrange(iters):
		pass
		#(X, y) = getTrainingVectors()
		#perceptron = perceptron.fit(X, y)

if __name__ == '__main__':
	main()