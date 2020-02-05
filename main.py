from geopandas import GeoDataFrame, sjoin
from shapely.geometry import Point
from sklearn.cluster import KMeans
import numpy as np

from object_detection.helper_functions import dataframe_to_laz
from object_detection.tree_detector import Detector_tree

tree = Detector_tree((122539.6, 490351.4, 122607.8, 490403.6))
tree.cluster_on_xy(40, 60)
tree.convex_hullify(points = tree.clustered_points)
# dataframe_to_laz(tree.clustered_points[tree.clustered_points['Classification'] >= 0], 'tst_fn.laz')


# for treeID in tree.tree_df.clusterID:
#     if treeID >= 0:
#         geometry = tree.tree_df.loc[treeID, 'geometry']
#         # find a big "tree"
#         if geometry.area >= 500:
#             # find the raw points within the 'tree'
#             points = [Point(coords) for coords in zip(cluster_points['X'], cluster_points['Y'], cluster_points['Z'])]
#
#             # kmeans cluster these points.


cluster_points = tree.raw_points[tree.groundmask]
xyz = [Point(coords) for coords in zip(cluster_points['X'], cluster_points['Y'], cluster_points['Z'])]
cluster_data = tree.preprocess(
    cluster_points[['X', 'Y', 'Z', 'Red', 'Green', 'Blue', 'Intensity', 'ReturnNumber', 'NumberOfReturns']]
)
points_df = GeoDataFrame(cluster_data, geometry=xyz)
pointInPolys = sjoin(points_df, tree.tree_df, how='left')
grouped_raw_points = pointInPolys.groupby('clusterID')['Red', 'Green', 'Blue', 'clusterID']

for name, group in grouped_raw_points:
    if group.shape[0] >= 2000:
        cluster_data = np.array([group.geometry.x,
                                 group.geometry.y]).T
                                 # group.geometry.z])
        kmeans = KMeans(n_clusters=5).fit(group)
        labels = kmeans.labels_
