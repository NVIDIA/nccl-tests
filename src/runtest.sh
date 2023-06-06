#!/bin/bash
# Convenience script to run all tests in the build output

for script in ../build/*_perf; do
	echo $script;
	./$script $@;
done
