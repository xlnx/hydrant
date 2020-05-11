#!/bin/sh

if [ "$#" -ne 1 ] || ! [ -f "$1" ]; then
	echo "Usage: $0 hostfile" >&2
	exit 1
fi

set -o xtrace

#mpirun --mca orte_base_help_aggregate 0 -n 1 hydra-zookeeper : --hostfile $1 -np $(cat $1 | wc -l) hydra-slave
mpirun --mca orte_base_help_aggregate 0 -n 1 hydra-zookeeper : --hostfile $1 -np 4 hydra-slave
