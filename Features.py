import itertools

def main():
	sFile = 'Data/europarl-v7-h102000-tok-leq40.de-en'
	aFile = 'Data/europarl-v7-h102000-40.de-en.gdfa'

	sentences = getSentences(sFile)
	alignments = getAlignments(aFile)
	targetOrder = getTargetOrder(sentences, alignments)

	global ind
	ind = 336 #314

	print sentences[ind][1]
	print sentences[ind][0]
	print ''
	print alignments[ind]
	print ''
	print targetOrder[ind]
	print [w for [i, w] in targetOrder[ind]]

def getSentences(sFile):
    sentences = []
    with open(sFile, 'rU') as sens:
        for line in sens:
            sentences.append(line.strip().split(' ||| '))
            if len(sentences) == 337: break
    return sentences

def getAlignments(aFile):
	alignments = []
	with open(aFile, 'rU') as aligns:
		for line in aligns:
			align = line.strip().split(' ')
			alignments.append([map(int, a.split('-')) for a in align])
			if len(alignments) == 337: break
	return alignments

def tarOrderToString(tarOrder):
	flatOrder = []
	for index in tarOrder:
		for word in index:
			flatOrder.append(word)
	return flatOrder

def getTargetOrder(sentences, alignments):
	targetOrder = []
	for i in xrange(len(sentences)):
		sSens = sentences[i][0].split(' ')
		aligns = alignments[i]
		tOrder = [[] for w in xrange(len(sentences[i][1].split(' ')))]
		for j in xrange(len(sSens)):
			tarIndex = getTargetIndex(j, aligns)
			tOrder[tarIndex].append([j, sSens[j]])
		targetOrder.append(tarOrderToString(tOrder))
	return targetOrder

def getTargetIndex(srcIndex, aligns):
	srcAligns = [tar for [src, tar] in aligns if src==srcIndex]
	# null alingment: join last alignment of previous src word
	if not srcAligns:
		if srcIndex == 0:
			srcAlign = 0
		else:
			for prev in reversed(xrange(srcIndex)):
				srcAlign = getTargetIndex(prev, aligns)
	# choose src alignment with highest tar index
	else:
		srcAlign = srcAligns[-1]
	return srcAlign

if __name__ == '__main__':
	main()
