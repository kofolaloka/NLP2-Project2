#!/bin/bash

corpusname=$1
src=$2
tar=$3
n=$4

parFileName="$corpusname.$src-$tar"
if [ ! -f $parFileName ]; then
    echo "creating parrallel file $parFileName"
    paste -d '|' "$parFileName.$src" /dev/null /dev/null "$parFileName.$tar" > $parFileName
else
    echo "reading existing parrallel file $parFileName"
fi


tokFileName="$corpusname-tok.$src-$tar"
if [ ! -f $tokFileName ]; then
    echo "creating lowercase tokenized corpus $tokFileName"
    ~/software/cdec/cdec-2014-10-12/corpus/tokenize-anything.sh < $parFileName | ~/software/cdec/cdec-2014-10-12/corpus/lowercase.pl > $tokFileName
else
    echo "reading existing lowercase tokenized corpus $tokFileName"
fi

filtFileName="$corpusname-tok-leq$n.$src-$tar"
#tokFileName="testfile"
if [ ! -f $filtFileName ]; then
    echo "filtering by sentence length $n, creating $filtFileName"
    cp $tokFileName $filtFileName 
    done=false
    while [ $done = false ]; do
        ~/software/cdec/cdec-2014-10-12/corpus/filter-length.pl -$n $tokFileName > $filtFileName # >& logfile
        error=$( grep -o "[[:digit:]]* / [[:digit:]]* : Corpus appears to be incorretly formatted, example" logfile | grep -o "[[:digit:]]*" | head -2 | tail -1 )
#        echo $error
        if [ "$error" = "" ]; then
            echo "no error"
            done=true
        else
            echo "Error at line line $error. Removing line and restarting filtering"
            newData=$( cat $tokFileName | sed "${error}d" ) # > testfile
            #echo head -5 "$test"
            echo "$newData" > $filtFileName
            wc -l $filtFileName
        fi
        sync
    done    
else
    echo "reading existing file filtered by sentence length $n: $filtFileName"
fi
echo "sentences left:"
wc -l $filtFileName

fwdFileName="$corpusname-$n.$src-$tar.fwd_align"
if [ ! -f $fwdFileName ]; then
    echo "creating src-tar alignment file $fwdFileName"
    ~/software/cdec/cdec-2014-10-12/word-aligner/fast_align -i $filtFileName -d -v -o > $fwdFileName
else
    echo "src-tar alignment file $fwdFileName already exists, skipping"
fi

revFileName="$corpusname-$n.$src-$tar.rev_align"
if [ ! -f $revFileName ]; then
    echo "creating tar-src alignment file $revFileName"
    ~/software/cdec/cdec-2014-10-12/word-aligner/fast_align -i $filtFileName -d -v -o -r > $revFileName
else
    echo "tar-src alignment file $revFileName already exists, skipping"
fi

symFileName="$corpusname-$n.$src-$tar.gdfa"
if [ ! -f $symFileName ]; then
    echo "Symmetrizing alignments, creating $symFileName"
    ~/software/cdec/cdec-2014-10-12/utils/atools -i $fwdFileName -j $revFileName -c grow-diag-final-and > $symFileName
else
    echo "$symFileName already exists, skipping"
fi
