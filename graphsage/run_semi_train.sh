#! /bin/bash

for ((i=1; i<=$1; i++)); 
do
	echo "Turns: "$i"/"$1;
	python semisupervised_train.py > "./out/"$i;
done


