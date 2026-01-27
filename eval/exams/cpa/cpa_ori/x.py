import pandas as pd
import os

for filename in os.listdir('./'):
    if filename.endswith('.csv'):
        basename = filename.replace('.csv','')
        df = pd.read_csv(filename)
        val_df = df.sample(frac=0.6,random_state=2023,axis=0)
        val_df.to_csv(f'../val/{basename}_val.csv',index=False)

        df = df[~df.index.isin(val_df.index)]
        dev_df = df.sample(n=5,random_state=2023,axis=0)
        dev_df['explanation'] = ''
        dev_df.to_csv(f'../dev/{basename}_dev.csv',index=False)

        df = df[~df.index.isin(dev_df.index)]
        df.to_csv(f'{filename}_train.csv',index=False)
