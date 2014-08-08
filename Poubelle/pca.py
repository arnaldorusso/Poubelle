"""
http://stats.stackexchange.com/questions/82050/principal-component-analysis-
\and-regression-in-python
"""

import pandas as pd
from sklearn.decomposition.pca import PCA

source = pd.read_csv('../files/multicollinearity.csv')
frame = pd.DataFrame(source)
cols = [col for col in frame.columns if col not in ['response']]
frame2 = frame[cols]

pca = PCA(n_components=5)
pca.fit(frame2)

# The amount of variance that each PC explains?
print pca.explained_variance_ratio_

# What are these? Eigenvectors?
print pca.components_

# Are these the eigenvalues?
print pca.explained_variance_

# it looks like sklearn won't operate directly on a pandas dataframe.
# Let's say that I convert it to a numpy array:

npa = frame2.values
npa

# If I then change the copy parameter of sklearn's PCA to False, it operates
# directly on the array `npa`
pca = PCA(n_components=5, copy=False)
pca.fit(npa)

print 'Explained Variance'
print pca.explained_variance_ratio_

print 'Components'
print pca.components_
