## Solving k-Nearest Neighbors with Math and Numpy<br>

### NOTE: Attached you can see the 'knn.py' file with the knn functions from scratch. The 'kNN_example.ipynb' file has an example with this implementation.


__k-Nearest Neighbors is a very commonly used algorithm for classification__. It works great when you have large amount of classes and a few samples per class, this is why it is very commonly used in __face recognition__.<br>
<br>
__kNN in one sentence__: is an algorithm that classifies and assigns labels based on the closest k-neighbors.<br>
<br>
k Parameter - Size of Neighborhood<br>
 - k represents the amount of neighbors to compare data with. That is why it usually k is an odd number.<br>
 - the bigger the k, the less 'defined' or more smooth are the areas of classification.<br>
<br>

__Distance__ is a key factor in order to determine who is the closest. Distance impacts the size and characteristics of the neighborhoods.  The most commonly used is Euclidean distance since it gives the closest distance between 2 points.<br>
<br>
Most Common Distances<br>
 - Euclidean: the shortest distance between to points that might not be the best option when features are normalized. Typically used in face recognition.<br>
 - Taxicab or Manhattan: is the sum of the absolute differences of the Cartesian coordinates of 2 points. It works the same way as when a car needs to move around 'blocks' to get to the destination.<br>
 - Minkowski: is a mix of both Euclidean and Mincowski.<br>

<br>
The amount of features impacts kNN significantly because the more points we have, the more 'unique' each neighborhood becomes. It also affects speed because we need to measure each distance first in order to determine who are the closest k neighbors.<br)

