# COOLnorm
This is a reference implementation of the paper Learning Graph Representations with Maximal Cliques


# Requirements
Python >= 3.6.x

1.15 <= Tensorflow < 2

# Run the code
python trainMain.py --dataset <cora|citeseer|pubmed> --model <COOL,COOLnorm> --alfa <?> --beta <?> 

Example: python trainMain.py --dataset cora --model COOLnorm --alfa 1 --beta 0

