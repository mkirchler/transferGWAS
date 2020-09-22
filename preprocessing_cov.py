import argparse
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input UKB covariate file')
    parser.add_argument('--indiv', type=str, help='Path to subset of individuals')
    parser.add_argument('--add_cov', nargs=2, action='append', help='Additional covariates as key-value pairs (as in `--add_cov 123-0.0 mytrait`)')
    parser.add_argument('--output', type=str, default='covariates.txt', help='Output filename')
    args = parser.parse_args()

    columns = {
            '31-0.0':       'sex',                  # sex
            '21022-0.0':    'age',                  # age at recruitment
            '54-0.0':       'assessment_center',    # assessment center
            '22000-0.0':    'geno_batch',           # genotyping batch
            # '20116-0.0',    # smoking status, self-reported touchscreen
            # '21000-0.0',    # ethnic background, self-reported touchscreen
            }

    for i in range(1, 41):
        columns[f'22009-0.{i}'] = f'genet_PC_{i}'

    if args.add_cov:
        for key, val in args.add_cov:
            columns[key] = val

    print('reading covariate file...')
    df = pd.read_csv(args.input, index_col=0, usecols=['eid']+list(columns.keys()))
    # df = pd.read_csv(args.input, nrows=10, index_col=0, usecols=['eid']+list(columns.keys()))
    print('preparing data...')
    df.rename(columns, axis=1, inplace=True)

    # binarize genotyping batch
    df.geno_batch = 1*(df.geno_batch > 0)

    if args.indiv:
        indiv = pd.read_csv(args.indiv, sep=' ', header=None)[0].values
        indiv_intersection = np.intersect1d(indiv, df.index)
        df = df.loc[indiv_intersection]

    df['FID'] = df.index
    df['IID'] = df.index
    cols = df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    df = df[cols]


    df.to_csv(args.output, sep=' ', index=False)
    
    
if __name__ == '__main__':
    main()
