from object_detection.helper_functions import dataframe_to_laz
from object_detection.tree_detector import Detector_tree



# losse bomen en clusters
box = 122539.6, 490351.4, 122607.8, 490403.6

# heel klein stukje
# box = 122544, 490380, 122553,490386
tree = Detector_tree((box))
tree.cluster_on_xy(40, 60)
tree.convex_hullify(points=tree.clustered_points)
# dataframe_to_laz(tree.clustered_points[tree.clustered_points['Classification'] >= 0], 'tst_fn.laz')


tree.merge_points_polygons()

tree.kmean_cluster()

write_df = tree.grouped_points[['X', 'Y', 'Z', 'Classification']]
dataframe_to_laz(write_df[write_df['Classification'] >= 0], 'tst_fn.laz')