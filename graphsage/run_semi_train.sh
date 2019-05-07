#! /bin/bash

echo "Training generator ..."
for ((i=1; i<=$1; i++)); 
do
	echo "Turns: "$i"/"$1;
	python semisupervised_train.py 1 3 6 > "./out/gen-"$i;
done

echo "Training discriminator ..."
for ((i=1; i<=$1; i++)); 
do
	echo "Turns: "$i"/"$1;
	python semisupervised_train.py 6 2 2 > "./out/dis-"$i;
done


