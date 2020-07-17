#!/usr/bin/env bash

# such as add cd /vandal/dataset
if [ $# -eq 1 ]; then
  dest=$1
else
  dest="."
fi

wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip -P $dest
unzip $dest/ADEChallengeData2016.zip

echo "Copy the files in data/ade_splits in the ade main folder."
cp data/ade_splits/* $dest