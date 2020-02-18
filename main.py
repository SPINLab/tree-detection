### local_maxima
from geopandas import GeoDataFrame
from sklearn.preprocessing import minmax_scale
from object_detection.helper_functions import dataframe_to_laz
from object_detection.tree_detector import DetectorTree



# losse bomen en clusters
box = 122539.6, 490351.4, 122607.8, 490403.6

# groter stukje
# box = 122577.4,483763.5 ,122720.1,483854.5

# stukje met daken
# box = 122262.1, 483733.3, 122425.3, 483844.7

# heel klein stukje
# box = 122544, 490380, 122553, 490386



tree = DetectorTree(box)
tree.cluster_on_xy(min_cluster_size=30, min_samples=10)
tree.convex_hullify(points=tree.clustered_points)
tree.df_to_PG(tree.tree_df, schema='bomen', table_name='xy_bomen')

tree.find_points_in_polygons(tree.tree_df)
tree.kmean_cluster(tree.xy_grouped_points, min_dist=2, min_height=0, gridsize=2)
tree.convex_hullify(tree.kmean_grouped_points)
tree.df_to_PG(tree.tree_df, schema='bomen', table_name='km_bomen')

write_df = tree.kmean_grouped_points[['pid', 'X', 'Y', 'Z', 'Red', 'Green', 'Blue']]
dataframe_to_laz(write_df, 'tst_fn.laz')



# radial density