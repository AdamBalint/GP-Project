#!/bin/bash
python RunGP.py 0 2 0 &
python RunGP.py 1 2 0 &
python RunGP.py 2 2 0 &
python RunGP.py 3 2 0 

wait

python RunGP.py 4 2 0 &
python RunGP.py 5 2 0 &
python RunGP.py 6 2 0 
wait
