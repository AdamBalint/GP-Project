#!/bin/bash
python RunGP.py 0 0 0 &
python RunGP.py 1 0 0 &
python RunGP.py 2 0 0 &
python RunGP.py 3 0 0 

wait

python RunGP.py 4 0 0 &
python RunGP.py 5 0 0 &
python RunGP.py 6 0 0 
wait
