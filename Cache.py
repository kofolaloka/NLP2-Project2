import os, os.path
import time
import datetime
import cPickle as pickle

class Cache(object):
    def __init__(self, fileName, storageObject, usePickle=False):
        self.cache = storageObject
        self.file = 'Cache/'+fileName
        self.usePickle = usePickle
        self.load()
        self.saveCache = True

    def load(self):
        try:
            print 'Reading cache %s...' % self.file
            with open(self.file, 'r') as f:
                try:
                    if self.usePickle:
                        self.cache = pickle.load(f)
                    else:
                        cache = f.read()
                        self.cache = eval(cache)
                except Exception, e:
                    print e
        except IOError:
            pass

    def save(self):
        if self.saveCache:
            print 'Saving cache %s...' % self.file
            start = time.time()
            if self.usePickle:
                try:
                    pickle.dump(self.cache, open(self.file, "w"))
                except IOError:
                    self.saveCache = False
                    print "Failed while writing cache. Cache will no longer be saved..."
            else:
                try:
                    with open(self.file, 'w') as f:
                        f.write(str(self.cache))
                except IOError:
                    self.saveCache = False
                    print "Failed while writing cache. Cache will no longer be saved..."
            print '\tDuration:', self.getDuration(start, time.time())

    def getDuration(self, start, stop):
        return str(datetime.timedelta(seconds=(stop-start)))