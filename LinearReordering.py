import numpy as np
from sklearn.linear_model import Perceptron
from itertools import izip, chain
from multiprocessing import Pool
from random import randint, sample
import time
import datetime
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix

templateFts = ['t[l-1]','w[l]','t[l]','t[l+1]','"-".join(t[l+1:r])','t[r-1]','w[r]','t[r]','t[r+1]']
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

	global cores
	cores = 8
	global templates
	templates = [[templateFts[t] for t in temp] for temp in templates]
	global perceptron
	perceptron = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, n_iter=1, shuffle=True, verbose=0, eta0=1.0,
                            n_jobs=1, random_state=0, class_weight=None, warm_start=False)

	# Prepare training data
	global sentences
	global tags
	sentences, tags, identityPerm = readTaggedData('Data/europarl-v7-h102000-tok-leq10.de-en.tagged.de')
	alignments = readAlignments('Data/europarl-v7-h102000-10.de-en.gdfa')
	global targetOrder
	targetOrder = getTargetOrder(sentences, alignments)

	test = len(sentences)
	idx = [i for i in range(len(sentences)) if len(sentences[i])>1]
	sentences = [sentences[i] for i in idx]
	tags = [tags[i] for i in idx]
	identityPerm = [identityPerm[i] for i in idx]
	targetOrder = [targetOrder[i] for i in idx]
	
	global features
	features = initFeatures()
	print '\t\t', len(features), 'features found'
	
	reorderingTraining(identityPerm, 1)

def readTaggedData(sFile):
	sentences = []
	tags = []
	identityPerm = []
	with open(sFile, 'rU') as sens:
		for line in sens:
			words = np.array([w.split('_') for w in line.strip().split(' ')])
			sentences.append(list(words[:,0]))
			tags.append(list(words[:,1]))
			identityPerm.append(range(len(words)))

			#break
			if len(sentences) == 8: break
	return sentences, tags, identityPerm 

def readAlignments(aFile):
	alignments = []
	with open(aFile, 'rU') as aligns:
		for line in aligns:
			align = line.strip().split(' ')
			alignments.append([map(int, a.split('-')) for a in align])
			if len(alignments) == 8: break
			#break
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
		targetOrder.append(list(tOrder[:,0]))
	return targetOrder

def reorder(source, order):
	return [source[i] for i in order]
    
def genPermutations(sen, n):
	print 'test'
	permutations = []
	while len(permutations) < n:
		perm = sorted([randint(0,len(sen)) for r in range(3)])
		if perm[0]!=perm[1] and perm[1]!=perm[2]:
			permutations.append(perm)
	print 'test 2'
	return permutations

def permute(sen, (i, j, k)):
	return sen[:i] + sen[j:k] + sen[i:j] + sen[k:]

def permuteNeighbor(sen, neighbor):
	if not 'perm' in neighbor:
		return sen
	newSen = sen
	if 'children' in neighbor:
		for child in neighbor['children']:
			newSen = permuteNeighbor(newSen, child)
	newSen = permute(newSen, neighbor['perm'])
	return newSen

def getTrainingVectors(sourceOrder):
	start = time.time()
	p = Pool(processes=cores)
	chunks = zip(getChunks(sentences, cores), getChunks(tags, cores),
		         getChunks(sourceOrder, cores), getChunks(targetOrder, cores))
	vectors = p.map(getTrainingVectorsProcess, chunks)	
	X = []
	y = []
	for Xi, yi in vectors:
		X += Xi
		y += yi		
	return csr_matrix(X), np.array(y)

def getTrainingVectorsProcess((sens, sTags, srcOrder, tarOrder)):
	X = []
	y = []
	for sen, tag, sOrder, tOrder in izip(sens, sTags, srcOrder, tarOrder):
		w = reorder(sen, sOrder)
		t = reorder(tag, sOrder)
		for l in xrange(len(sen)):
			for r in xrange(1, len(sen)):
				sample = getSample(w, t, l, r)
				X.append(sample)
				if tOrder.index(sOrder[l]) < tOrder.index(sOrder[r]):
					y.append(0) # don't invert
				else:
					y.append(1) # invert
	return X, y

def getFeatureValue(template, w, t, l, r):
	value = ''
	for part in template:
		if 'l-1' in part and not l>0:
			continue
		if 'r-1' in part and not r>0:
			continue	
		if 'l+1' in part and not l<len(w)-2:
			continue
		if 'r+1' in part and not r<len(w)-2:
			continue
		if 'b' in part and not r-l<5:
			continue
		try:
			result = eval(part)
		except IndexError:
			print part
			print w
			print t
			print l
			print r
		value += '_'+result
	return value[1:]

def getExtendedValue(featureValue, l, r):
	d = r-l
	if d > 10:
		d = '>10'
	elif d > 5:
		d = '>5'
	return featureValue+'_'+str(d)

def getChunks(array, c):
    n = len(array)
    minSize = n // c
    remainder = n % c
    if c > n:
        raise Exception("getChunks called with number of chunks greater than length of array")
    chunks = []
    nextElement=0
    for i in xrange(c):
        if i < remainder:
            size = minSize + 1
        else:
            size = minSize
        chunks.append(array[nextElement:nextElement+size])
        nextElement = nextElement+size
    return chunks

def initFeatures():
	start = time.time()
	p = Pool(processes=cores)
	chunks = zip(getChunks(sentences, cores), getChunks(tags, cores))
	feats = p.map(getFeaturesProcess, chunks)
	print '\tInitialized features in', getDuration(start, time.time())				
	return list(set(chain(*feats)))

def getFeaturesProcess((sens, senTags)):
	feats = []
	for sen, tag in izip(sens, senTags):
		for l in xrange(len(sen)):
			for r in xrange(l+1, len(sen)):
				for tem in templates:
					value = getFeatureValue(tem, sen, tag, l, r)
					feats.append(value)
					feats.append(getExtendedValue(value, l, r))
	return feats


def localSearch(sourceOrder):
	start = time.time()
	p = Pool(processes=cores)
	chunks = zip(getChunks(sentences, cores), 
				 getChunks(tags, cores),
				 getChunks(sourceOrder, cores), 
				 getChunks(targetOrder, cores))
	neighbors = p.map(localSearchProcess, chunks)
	print '\tFound best neighbors in', getDuration(start, time.time())
	return list(chain(*neighbors))

def localSearchProcess((sens, senTags, srcOrder, tarOrder)):
	neighbors = []
	for sen, tag, sOrder, tOrder in izip(sens, senTags, srcOrder, tarOrder):
		n = len(sen)
		beta = {(i,i+1):{'value':0} for i in xrange(0,n)}
		delta = {}
		for w in xrange(2, n+1):
			for i in xrange(0, n-w+1):
				k = i+w
				beta[i,k] = -float('inf')
				for j in xrange(i+1, k):
					delta = addBenefit(sOrder, i, j, k, tOrder, sen, tag, delta)
					senBeta = beta[i,j]['value'] + beta[j,k]['value'] + max(0, delta[i,j,k])
					if senBeta > beta[i,k]:
						children = [beta[i,j], beta[j,k]]
						beta[i,k] = {'value': senBeta, 'perm':(i,j,k), 'children': children}
		neighbors.append(beta[0,n])
	return neighbors

def getPreference(sOrder, l, r, sen, tag):
	w = reorder(sen, sOrder)
	t = reorder(tag, sOrder)
	sample = getSample(w, t, l, r)
	return perceptron.predict(csr_matrix([sample]))[0]

def getSample(w, t, l, r):
	sample = [0]*len(features)
	for tem in templates:
		value = getFeatureValue(tem, w, t, l, r)
		try:
			sample[features.index(value)] = 1
		except ValueError:
			pass
		else:
			extended = getExtendedValue(value, l, r)
			try:
				sample[features.index(extended)] = 1
			except ValueError:
				pass
	return sample

def addBenefit(sOrder, i, j, k, tOrder, sen, tag, delta):
	if (i,j,k) in delta:
		return delta
	if i is j or j is k:
		delta[i,j,k] = 0
		return delta
	
	delta = addBenefit(sOrder, i, j, k-1, tOrder, sen, tag, delta)
	delta = addBenefit(sOrder, i+1, j, k, tOrder, sen, tag, delta)
	delta = addBenefit(sOrder, i+1, j, k-1, tOrder, sen, tag, delta)

	benefit = delta[i,j,k-1] + delta[i+1,j,k] - delta[i+1,j,k-1] 
	benefit += getPreference(sOrder, k-1, i, sen, tag)
	benefit -= getPreference(sOrder, i, k-1, sen, tag)
	delta[i,j,k] = benefit

	return delta

def reorderingTraining(identityPerm, iters=10):
	global perceptron 
	sourceOrder = identityPerm

	print 'Beginning training...'
	tStart = time.time()
	X, y = getTrainingVectors(sourceOrder)
	perceptron = perceptron.fit(X, y)
	print 'Perceptron initialized in', getDuration(tStart, time.time())
	
	iStart = time.time()
	for i in xrange(1, iters+1):
		print '\tITERATION', i
		bestNeighbors = localSearch(sourceOrder)
		newSource = [permuteNeighbor(sourceOrder[i], bestNeighbors[i]) for i in xrange(len(sourceOrder))]
		vStart = time.time()
		X, y = getTrainingVectors(newSource)
		perceptron = perceptron.fit(X, y)
		print '\tperceptron updated in', getDuration(vStart, time.time())
		sourceOrder = newSource
		print '\tDuration:', getDuration(iStart, time.time())

	s = 0
	test = localSearch(identityPerm)
	print sentences[s]
	print test[s]
	print reorder(sentences[s], targetOrder[s])
	print [sentences[s][i] for i in permuteNeighbor(identityPerm[s], test[s])]
	source = [permuteNeighbor(identityPerm[i], test[i]) for i in xrange(len(identityPerm))]
	print identityPerm[s]
	print targetOrder[s]

	print 'Duration:', getDuration(tStart, time.time())

def getDuration(start, stop):
    return str(datetime.timedelta(seconds=(stop-start)))

if __name__ == '__main__':
	main()
