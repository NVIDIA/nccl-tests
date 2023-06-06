#!/bin/bash
# Convenience script to run all tests in the build output

for script in ../build/*_perf; do
	echo $script;
	if [ -z "$@"]; then
		./$script -b 8 -e 128M -f 2 -g 2 #convenient default, tests a variety of loads
	else 
		./$script $@;
	fi
done
