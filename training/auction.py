__author__ = 'bptripp'

"""
Bertsekas' Auction Algorithm. This is a nearly line-by-line port of the Matlab implementation by Florian Bernard:

    http://www.mathworks.com/matlabcentral/fileexchange/48448-fast-linear-assignment-problem-using-auction-algorithm

And this is a nice introduction to the algorithm:

    Bertsekas, D. P. (1990). The Auction Algorithm for Assignment and Other Network Flow Problems: A Tutorial.
    Interfaces, 20(4)

Bernard's Matlab code is distributed under the BSD License, as follows:

    Copyright (c) 2014, Florian Bernard
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in
          the documentation and/or other materials provided with the distribution

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

This code is also distributed under the BSD License, Copyright (2015) Bryan Tripp.

Note that Bertsekas's Fortran code is also online: http://www.mit.edu/~dimitrib/auction.txt
"""

import time
import numpy as np

def auction(A, epsilon=None, epsilonDecreaseFactor=None):
    """
    Parameters
    ----------
    :param A:
    :param epsilon:
    :param epsilonDecreaseFactor:

    Returns
    -------
    :return assignments:
    :return prices:
    """

    N = A.shape[0]
    prices = np.ones(N)

    A = A * N+1

    # heuristic for setting epsilon
    if epsilon is None:
        # epsilon = np.max(np.abs(A[:])) / 25.0 #this sometimes make us skip the loop
        epsilon = np.maximum(1, np.max(np.abs(A[:])) / 15.0)

    if epsilonDecreaseFactor is None:
        epsilonDecreaseFactor = 0.2

    while epsilon >= 1:
        # The outer loop performs epsilon-scaling in order to speed up execution
        # time. In particular, an updated prices array is computed in each
        # round, which speeds up further rounds with lower values of epsilon.
        assignments = np.empty(N)
        assignments[:] = np.NAN

        while np.any(np.isnan(assignments)):
            # Forward-Auction Algorithm -- Bidding Phase

            # find unassigned indices
            unassignedIdx, = np.nonzero(np.isnan(assignments))
            nUnassigned = len(unassignedIdx)

            # find best and second best objects
            AijMinusPj = A[unassignedIdx, :] - prices

            viIdx = np.argmax(AijMinusPj, axis=1)

            for i in range(nUnassigned):
                AijMinusPj[i][viIdx[i]] = -np.Inf

            # print(AijMinusPj)

            wi = np.max(AijMinusPj, axis=1)

            # compute bids
            bids = np.empty(nUnassigned)
            bids[:] = np.NAN
            for i in range(nUnassigned):
                bids[i] = A[unassignedIdx[i], viIdx[i]] - wi[i] + epsilon

            # Assignment Phase
            objectsThatHaveBeenBiddedFor = np.unique(viIdx)

            for uniqueObjIdx in range(len(objectsThatHaveBeenBiddedFor)):
                currObject = objectsThatHaveBeenBiddedFor[uniqueObjIdx]
                personssWhoGaveBidsForJ = np.nonzero(viIdx==currObject)

                b = bids[personssWhoGaveBidsForJ]
                idx = np.argmax(b)
                prices[currObject] = b[idx]
                personWithHighestBid = unassignedIdx[personssWhoGaveBidsForJ[0][idx]]

                # remove previous assignment and store new assignment (person with highest bid)
                assignments[assignments==currObject] = np.NaN
                assignments[personWithHighestBid] = currObject
                # print(assignments)

        epsilon = epsilon * epsilonDecreaseFactor # refine epsilon

    return assignments, prices

if __name__ == '__main__':
    A = np.array([[5, 9, 2], [10, 3, 2], [8, 7, 4]])
    print(A)
    start_time = time.time()
    assignments, prices = auction(A)
    print(assignments)
    print('auction time: ' + str(time.time() - start_time))
