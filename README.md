# array_prevalence
Provides python script to estimate prevalence from array testing data.

This script estimates the posterior for prevalence given a set of array testing results. The path to the input data is provided via the -i flag. 
Input data should be a csv file with two columns, the first indicating the number of positive row tests and the second the number of positive column tests (or vice versa), with no headers, i.e.
            1,2
            1,1
            0,1
            3,2
            ...

Users can choose between MLE estimate of the prevalence (faster default) and obtaining the full posterior distribution (-p flag).

## Installation

Download estimate_prevalence_array.py and make sure the following packages are installed:
    sys
    json
    random
    argparse
    numpy
    pandas
    math
    operator
    functools
    scipy

## Usage

Go to the directory where estimate_prevalence_array.py is stored. An example to obtain the MLE prevalence estimate with the datastored in the same path as array_test_results.csv for array testing with 36 samples in each group testing design and estimated specificity of 0.99 and sensitivity of 0.98 is

```
python ./estimate_prevalence_array.py -i ./array_test_results.csv -n 35 -s 0.98 -z 0.99 
```

## Arguments
        -i  Path to data csv
        -n  Number of samples in each group testing design
        -s  Expected sensitivity (default = 0.99 for group size of 8 or less, and 0.95 for more than 8, i.e. n <= 64 and n > 64 respectively)
        -z  Expected specificity (default = 0.99)  
        -o  Output filename for posterior dataframe (default './')
        -p  Flag that when provided enables full posterior estimation (default: off, computing only MLE prevalence estimate)

## Citation

TO FOLLOW
