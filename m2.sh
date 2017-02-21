#!/bin/bash
python RunGP.py 0 1 0 &
python RunGP.py 1 1 0 &
python RunGP.py 2 1 0 &
python RunGP.py 3 1 0 

wait

python RunGP.py 4 1 0 &
python RunGP.py 5 1 0 &
python RunGP.py 6 1 0 
wait
