import os
import glob
import pandas as pd
MY_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(MY_DIR)

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "combined.csv", index=False, encoding='utf-8-sig')

