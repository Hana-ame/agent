#!/bin/bash
# Back up the correct mechanism/step configs before running the builder,
# and provide a restore script. The builder regenerates configs with
# wrong assumptions (bias 0.5, no mechanism params), so we restore ours.
set -e
CFG=additive-rand-transformer/configs
BK=config_backup
rm -rf $BK && mkdir -p $BK
cp $CFG/*.json $BK/
echo "backed up $(ls $BK/*.json | wc -l) configs to $BK/"
