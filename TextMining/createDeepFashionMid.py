import pandas as pd
import numpy as np
import constants


#Save to .pkl middle tables of deepfashion as they had many rows with possible values

labelsDEEP = pd.read_csv(constants.DEEPFASHIONPATH, sep=r"[ ]{2,}", skiprows = 1, engine = 'python')
deepfashionNamesDF = pd.DataFrame({'attribute_type':range(1,6), 'attribute_group_name':constants.DEEPFASHIONATTRIBUTES})
deepfashionLabelsDF = pd.merge(labelsDEEP, deepfashionNamesDF, on = 'attribute_type').replace(' ', np.nan).dropna()
groups = deepfashionLabelsDF.groupby('attribute_group_name')['attribute_name']


for attribute in constants.DEEPFASHIONATTRIBUTES:
    what = groups.get_group(str(attribute)).reset_index()
    what.loc[:,'index'] = what.index
    what.loc[:,'index'] = 1 + what.loc[:,'index']
    what = pd.concat([what, what.iloc[:,0]], axis = 1, ignore_index= True)
    what.columns = ['id', 'value', 'generalid']
    what['value'] = what['value'].str.upper()
    what = what.astype({'id':'int32','value':'string','generalid':'int32'})
    # df = df.append(what)
    # print(df.dtypes)
    #print(what)
    what.to_pickle('/home/alexandros/Desktop/'+ attribute +'.pkl')
