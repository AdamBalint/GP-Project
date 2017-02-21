#!/bin/bash
python RunGP.py 0 3 0 &
python RunGP.py 1 3 0 &
python RunGP.py 2 3 0 &
python RunGP.py 3 3 0 

wait

python RunGP.py 4 3 0 &
python RunGP.py 5 3 0 &
python RunGP.py 6 3 0 
wait
