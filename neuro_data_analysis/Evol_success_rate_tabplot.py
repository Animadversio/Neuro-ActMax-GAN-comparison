

import scipy
import scipy.special


n = 20
k = 17
q = 0.5  # median
scipy.special.betaincinv(k+1, n+1-k, q)
# Gives a success rate of 83% (71% - 91%).