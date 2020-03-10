### local_maxima
from geopandas import GeoDataFrame
from shapely.geometry import Point

from object_detection.helper_functions import df_to_pg, dataframe_to_laz, get_colors
from object_detection.tree_detector import DetectorTree
import numpy as np

# losse bomen en clusters
# box = 122539.6, 490351.4, 122607.8, 490403.6

# groter stukje
# box = 122577.4, 483763.5, 122720.1, 483854.5

# stukje met daken
# box = 121970.5, 483782.7, 122107.5, 483871.9

# heel groot stuk met daken!
# box = 121898.8, 483741.2, 122172.7, 483919.5

# heel klein stukje
# box = 122544, 490380, 122553, 490386

# stammen
box = 122287.4, 483709.0, 122398.8, 483781.5

# GROOOOOOOT ring a 10 stuk? misschien wel grootst mogelijk....
# box = 125889.8,489393.5 , 126516.1,489791.3

# homogene bomenrij
# box = 126056.9,489631.9, 126132.1,489698.9

# klein stukje a10
# box = 126014.7, 489644.0, 126055.8,489680.6

# dak
# box = 122317.5,483749.3, 122443.9,483838.5

# Stukje kerngis
box = 123727.2, 482705.0, 123949.8, 482846.3

# defining the object that holds the points, tha mask and more
tree = DetectorTree(box)
tree.hdbscan_on_points(min_cluster_size=30, min_samples=10, xyz=False)
# tree.hdbscan_on_points(min_cluster_size=30, min_samples=10, xyz=False)
tree.convex_hullify(points=tree.clustered_points)
df_to_pg(tree.tree_df, schema='bomen', table_name='xy_bomen')

tree.find_points_in_polygons(tree.tree_df)
tree.kmean_cluster(tree.xy_grouped_points)

tree.convex_hullify(tree.kmean_grouped_points, kmean_pols=True)

df_to_pg(tree.tree_df, schema='bomen', table_name='km_bomen')

tree.tree_coords['geometry'] = [Point(x, y) for x, y, z in zip(tree.tree_coords.X,
                                                               tree.tree_coords.Y,
                                                               tree.tree_coords.Z)]
tree.tree_coords = GeoDataFrame(tree.tree_coords, geometry='geometry')

df_to_pg(tree.tree_coords, schema='bomen', table_name='stammen')


# colors/visualizing
colors = get_colors(len(np.unique(tree.kmean_grouped_points['Classification'])))
write_df = tree.kmean_grouped_points[['pid',
                                      'X', 'Y', 'Z',
                                      # 'Red', 'Green', 'Blue',
                                      'ReturnNumber', 'Classification']]

for i, color in enumerate(['Red', 'Green', 'Blue']):
    col = write_df.apply(lambda row: colors[int(row['Classification'])][i], axis=1)
    write_df.loc[:, color] = col

dataframe_to_laz(write_df, 'tst_fn.laz')
