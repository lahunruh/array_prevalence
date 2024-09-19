import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
from math import comb
from operator import mul
from functools import reduce
from scipy.stats import binom

def get_count(i,j,d):

    count = np.empty(shape=(i+1, j+1), dtype='object')
    if i*j >=1:
        count[0,0] = 0
        count[1,0] = 0
        count[0,1] = 0
    else:
        return 1

    for k in range(0,i+1):
        for l in range(0,j+1):
            if k == 0 and l == 0:
                continue
            if k == 0 and l == 1:
                continue
            if k == 1 and l == 0:
                continue
            if (k-1) >= 0:
                c1 = comb(k, k - 1) * comb(l, l) * count[k - 1, l]
            else:
                c1 = 0
            if (l-1) >= 0:
                c2 = comb(k, k) * comb(l, l - 1) * count[k, l - 1]
            else:
                c2 = 0
            count[k,l] = comb(k*l,d) - sum(comb(k,a) * comb(l,b) * count[a,b] for a in range(k+1) for b in range(l+1) if a < k or b < l) #- c1 - c2

    return count[i,j]

def get_ll(i,j,a,b,e1,f1,e2,f2,n,d,Se,Sp,all_counts):
    comb1 = comb((int(np.sqrt(n)) - a) + (int(np.sqrt(n)) - b), e1 + f1) * (1 - Sp)**(e1+f1) * Sp**(2*np.sqrt(n) - a - b - e1 - f1)
    comb2 = comb(a+b,e2+f2) * (1-Se)**(e2+f2) * Se**(a+b-e2-f2)
    comb3 = comb(int(np.sqrt(n)), a) * comb(int(np.sqrt(n)), b)
    count = all_counts[a][b][d]
    posscomb = comb(n,d)
    return comb1 * comb2 * comb3 * count / posscomb

def est_preval(data,n, store,Se,Sp,all_counts):

    preval = np.arange(0,1,0.001)
    precomp={}
    for p in preval:
        precomp[p] = []
        for i in range(len(data)):
            if (data[i][0],data[i][1],p) not in store:
                ll = sum(get_ll(data[i][0],data[i][1],a,b,e1,f1,e2,f2,n,d,Se,Sp,all_counts) * binom.pmf(d,n,p) for d in range(0,n+1) for a in range(0,min(d+1,int(np.sqrt(n)+1))) for b in range(0,min(d+1,int(np.sqrt(n)+1))) for e1 in range(0,int(np.sqrt(n)) - a + 1) for f1 in range(0,int(np.sqrt(n)) - b + 1) for e2 in range(0,a+1) for f2 in range(0,b+1) if (e1 - e2) == (data[i][0] - a) if (f1 - f2) == (data[i][1] - b) if a*b >= d)
                store[(data[i][0],data[i][1],p)] = ll
                store[(data[i][1],data[i][0],p)] = ll
            else:
                ll = store[(data[i][0],data[i][1],p)]
            precomp[p].append(ll)
    tot = 0
    for k in precomp:
        tot += reduce(mul,precomp[k],1) # * 1/len(preval)
    probs = []
    tt = 0
    for k in precomp:
        if tot > 0:
            probs.append((reduce(mul,precomp[k],1))/tot) #  * 1/len(preval))/tot)
        else:
            probs.append(0)
    df = pd.DataFrame([probs,preval]).T
    df.columns = ['prob','prevalence']
    return df, preval[probs.index(max(probs))]

def est_preval_MLE(data,n, store,Se,Sp,all_counts):

    preval = np.arange(0,1,0.001)
    precomp={}
    sdata = 0
    for i in range(len(data)):
        sdata += data[i][0]*data[i][1]/4
    sdata = sdata/len(data)
    sdata = sdata/n
    oldp=0.0
    for pidx, pt in enumerate(preval):
        if pt > sdata:
            p = oldp
            p_idx = max([0,pidx - 1])
            break
        oldp=pt
    precomp = []
    for i in range(len(data)):
        if (data[i][0],data[i][1],p) not in store:
            ll = sum(get_ll(data[i][0],data[i][1],a,b,e1,f1,e2,f2,n,d,Se,Sp,all_counts) * binom.pmf(d,n,p) for d in range(0,n+1) for a in range(0,min(d+1,int(np.sqrt(n)+1))) for b in range(0,min(d+1,int(np.sqrt(n)+1))) for e1 in range(0,int(np.sqrt(n)) - a + 1) for f1 in range(0,int(np.sqrt(n)) - b + 1) for e2 in range(0,a+1) for f2 in range(0,b+1) if (e1 - e2) == (data[i][0] - a) if (f1 - f2) == (data[i][1] - b) if a*b >= d)
            store[(data[i][0],data[i][1],p)] = ll
            store[(data[i][1],data[i][0],p)] = ll
        else:
            ll = store[(data[i][0],data[i][1],p)]
        precomp.append(ll)
    MLE_init = reduce(mul,precomp)
    pold = p
    max_MLE = 0
    max_p = 0
    while True:
        if p > 0:
            p = pold - 0.001
            precomp = []
            for i in range(len(data)):
                if (data[i][0],data[i][1],p) not in store:
                    ll = sum(get_ll(data[i][0],data[i][1],a,b,e1,f1,e2,f2,n,d,Se,Sp,all_counts) * binom.pmf(d,n,p) for d in range(0,n+1) for a in range(0,min(d+1,int(np.sqrt(n)+1))) for b in range(0,min(d+1,int(np.sqrt(n)+1))) for e1 in range(0,int(np.sqrt(n)) - a + 1) for f1 in range(0,int(np.sqrt(n)) - b + 1) for e2 in range(0,a+1) for f2 in range(0,b+1) if (e1 - e2) == (data[i][0] - a) if (f1 - f2) == (data[i][1] - b) if a*b >= d)
                    store[(data[i][0],data[i][1],p)] = ll
                    store[(data[i][1],data[i][0],p)] = ll
                else:
                    ll = store[(data[i][0],data[i][1],p)]
                precomp.append(ll)
            MLE_upper = reduce(mul,precomp)
        else:
            MLE_upper = 0
        p = pold + 0.001
        precomp = []
        for i in range(len(data)):
            if (data[i][0],data[i][1],p) not in store:
                ll = sum(get_ll(data[i][0],data[i][1],a,b,e1,f1,e2,f2,n,d,Se,Sp,all_counts) * binom.pmf(d,n,p) for d in range(0,n+1) for a in range(0,min(d+1,int(np.sqrt(n)+1))) for b in range(0,min(d+1,int(np.sqrt(n)+1))) for e1 in range(0,int(np.sqrt(n)) - a + 1) for f1 in range(0,int(np.sqrt(n)) - b + 1) for e2 in range(0,a+1) for f2 in range(0,b+1) if (e1 - e2) == (data[i][0] - a) if (f1 - f2) == (data[i][1] - b) if a*b >= d)
                store[(data[i][0],data[i][1],p)] = ll
                store[(data[i][1],data[i][0],p)] = ll
            else:
                ll = store[(data[i][0],data[i][1],p)]
            precomp.append(ll)
        MLE_lower = reduce(mul,precomp)
        if MLE_upper > max_MLE:
            max_MLE=MLE_upper
            max_p = pold - 0.001
        if MLE_lower > max_MLE:
            max_MLE=MLE_lower
            max_p = pold + 0.001
        if MLE_upper < MLE_init and MLE_lower < MLE_init:
            break
        else:
            MLE_init = max([MLE_upper,MLE_lower])
            if MLE_upper > MLE_lower:
                pold = pold - 0.001
            else:
                pold = pold + 0.001
    return max_p


if __name__ == '__main__':

    '''
    This script estimates the posterior for prevalence given a set of array testing results. The path to the input data is provided via the -i flag. 
    Input data should be a csv file with two columns, the first indicating the number of positive row tests and the second the number of positive column tests (or vice versa), with no headers, i.e.
            1,2
            1,1
            0,1
            3,2
            ...
    Options are:
        -i  Path to data csv
        -n  Number of samples in each group testing design
        -s  Expected sensitivity (default = 0.99 for group size of 8 or less, and 0.95 for more than 8, i.e. n <= 64 and n > 64 respectively)
        -z  Expected specificity (default = 0.99)  
        -o  Output filename for posterior dataframe
        -p  Flag that when provided enables full posterior estimation (default off, computing only MLE prevalence estimate)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', type=str, required=True, help='Path to data.')
    parser.add_argument('-n', dest='n',type=int,required=True, help='Number of samples in array testing design.')
    parser.add_argument('-s', dest='sensitivity',type=float, default=0, required=False, help='Expected sensitivity (default = 0.99 for group size of 8 or less, and 0.95 for more than 8, i.e. n <= 64 and n > 64 respectively)')
    parser.add_argument('-z', dest='specificity',type=float, default=0.99, required=False, help='Expected specificity (default = 0.99)')
    parser.add_argument('-o', dest='outpath',type=str, default='./',required=False, help='Path for writing posterior csv')
    parser.add_argument('-p', dest='posterior',action='store_true',help='If provided conducts full posterior estimation')
    parser.set_defaults(append=False)
    args = parser.parse_args()
    n = args.n
    if args.sensitivity == 0:
        if n <= 64:
            Se = 0.99
        else:
            Se = 0.95
    else:
        Se = args.sensitivity
    Sp=args.specificity
    data_df = pd.read_csv(args.input,header=None)
    samples = data_df.shape[0]
    
    data = []
    for i in range(data_df.shape[0]):
        data.append((data_df.iloc[0,i],data_df.iloc[1,i]))

    all_results = {}
    store = {}

    print('Getting All Counts')
    all_counts = {}
    for i in range(int(2*np.sqrt(n))+1):
        print(i,'of',2*np.sqrt(n))
        all_counts[i] = {}
        for j in range(int(2*np.sqrt(n))+1):
            all_counts[i][j] = {}
            for d in range(n+1):
                all_counts[i][j][d] = get_count(i,j,d)
                
    if args.posterior:
        posterior_df, p = est_preval(data,n,store,Se,Sp, all_counts)
        print('Prevalence is estimated at',p)
        print('Writing Posterior to ',args.outpath)
        sys.stdout.flush()
        posterior_df.to_csv(args.outpath, index=False)
    else:
        p = est_preval_MLE(data,n,store,Se,Sp, all_counts)
        print('Prevalence is estimated at',p)
        sys.stdout.flush()
        
        
