import pandas as pd 
import glob 

pd.concat([pd.read_csv(f) for f in glob.glob("*.csv")],
           axis =0, ignore_index=True).to_csv("merged.csv", index=False)