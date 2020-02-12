### local_maxima
from geoalchemy2 import WKTElement, Geometry
from sqlalchemy import create_engine

from object_detection.helper_functions import dataframe_to_laz
from object_detection.tree_detector import Detector_tree



# losse bomen en clusters
# box = 122539.6, 490351.4, 122607.8, 490403.6

# groter stukje
box = 122428.9,490392.9 ,122700.4,490564.7

# heel klein stukje
# box = 122544, 490380, 122553,490386
tree = Detector_tree(box)
tree.cluster_on_xy(40, 60)
tree.clustered_points['value_clusterID'] = 0
tree.convex_hullify(points=tree.clustered_points)
tree.df_to_PG(tree.tree_df, schema='bomen', table_name='xy_bomen')
# dataframe_to_laz(tree.clustered_points[tree.clustered_points['Classification'] >= 0], 'tst_fn.laz')

tree.find_points_in_polygons()
tree.kmean_cluster(tree.xy_grouped_points)
tree.convex_hullify(tree.kmean_grouped_points)
tree.df_to_PG(tree.tree_df, schema='bomen', table_name='km_bomen')

write_df = tree.kmean_grouped_points[['X', 'Y', 'Z', 'Classification']]
dataframe_to_laz(write_df[write_df['Classification'] >= 0], 'tst_fn.laz')


