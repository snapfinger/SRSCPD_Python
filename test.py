from utils import *

# test for matricize
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = 2 * a
c = 3 * a
d = np.stack([a, b, c], axis=2)
d_mat = matricize(d, 1)
print(d_mat)
# d_mat should be
# [[ 1  4  7  2  8 14  3 12 21]
#  [ 2  5  8  4 10 16  6 15 24]
#  [ 3  6  9  6 12 18  9 18 27]]
