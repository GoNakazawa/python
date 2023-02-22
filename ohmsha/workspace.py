import pandas as pd


dataset = pd.read_csv('iris_without_species.csv', index_col=0)

basic_statics = pd.concat([dataset.min(),dataset.median(),dataset.max(),dataset.mean(),dataset.var(),dataset.std()],
                          axis=1).T
basic_statics.index = ['min', 'median', 'max', 'mean', 'ver', 'std']
basic_statics.to_csv("gota.csv")