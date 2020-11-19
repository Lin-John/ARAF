import pandas as pd


def load_data_ccu():
    fp_name = r"data/ccu/names.txt"
    names = []
    with open(fp_name, 'r') as f:
        for line in f.readlines():
            names.append(line.strip())
    df = pd.read_csv(r"data/ccu/CommViolPredUnnormalizedData.txt", names=names)
    df=df[~df['ViolentCrimesPerPop'].isin(['?'])]
    for f in ['murders', 'rapes', 'robberies', 'assaults']:
        df.loc[:,f] = df[f].astype('int')
    df.loc[:,'ViolentCrimesPerPop'] = df['ViolentCrimesPerPop'].astype('float')
    df = df[df['murders']+df['rapes']+df['robberies']+df['assaults'] < 1.25/100000*df['ViolentCrimesPerPop']*df['population']]
    df = df[df['murders']+df['rapes']+df['robberies']+df['assaults'] > 0.8/100000*df['ViolentCrimesPerPop']*df['population']]
    droped = ['state', 'communityname', 'countyCode', 'communityCode', 'fold',
              'murders', 'murdPerPop', 'rapes', 'rapesPerPop', 'robberies',
              'robbbPerPop', 'assaults', 'assaultPerPop', 'burglaries', 'burglPerPop',
              'larcenies', 'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons',
              'arsonsPerPop', 'nonViolPerPop']
    columns = []
    df = df.drop(droped, axis=1)
    for f in df.columns:
        if not any(df[f].isin(['?'])):
            columns.append(f)
    return df[columns], columns[:-1], [], columns[-1:], [], 'regression'
