### local_maxima
from object_detection.helper_functions import df_to_pg, dataframe_to_laz
from object_detection.tree_detector import DetectorTree

# losse bomen en clusters
# box = 122539.6, 490351.4, 122607.8, 490403.6

# groter stukje
# box = 122577.4,483763.5 ,122720.1,483854.5

# stukje met daken
# box = 121970.5, 483782.7, 122107.5, 483871.9

# heel groot stuk met daken!
# box = 121898.8, 483741.2, 122172.7, 483919.5

# heel klein stukje
# box = 122544, 490380, 122553, 490386

# stammen
box =122287.4, 483709.0, 122398.8, 483781.5

# dak
# box = 122317.5,483749.3, 122443.9,483838.5

tree = DetectorTree(box)
tree.cluster_on_xy(min_cluster_size=30, min_samples=10)
tree.convex_hullify(points=tree.clustered_points)
df_to_pg(tree.tree_df, schema='bomen', table_name='xy_bomen')

# tree.find_points_in_polygons(tree.tree_df)
# group = tree.find_stems(tree.xy_grouped_points) # , grid_size=1, min_dist=1)
# df_to_pg(tree.tree_coords, schema='bomen', table_name='stammen')

tree.find_points_in_polygons(tree.tree_df)

tree.kmean_cluster(tree.xy_grouped_points, min_dist=5, min_height=0, gridsize=3)
tree.convex_hullify(tree.kmean_grouped_points)
df_to_pg(tree.tree_df, schema='bomen', table_name='km_bomen')
#
# write_df = tree.kmean_grouped_points[['pid', 'X', 'Y', 'Z', 'Red', 'Green', 'Blue']]
# dataframe_to_laz(write_df, 'tst_fn.laz')

# radial density