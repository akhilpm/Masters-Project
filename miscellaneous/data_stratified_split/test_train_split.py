# test train split

import random
lines = open('vehicle.dat').readlines()
test_size = int(len(lines)*.3)
random.shuffle(lines)

open('test.dat', 'w+').writelines(lines[0 : test_size])
open('train.dat', 'w+').writelines(lines[test_size : len(lines)])

