import psycopg2

from object_detection.helper_functions import df_to_pg, dataframe_to_laz, color_clusters, execute_query
from object_detection.tree_detector import DetectorTree

# losse bomen en clusters
# box = 122539.6, 490351.4, 122607.8, 490403.6

# test area for refactored code
kerngis_test_area = 'POLYGON((125935.34666449 484750.6,125938.349100001 484781.937500001,125935.5211 484849.719500003,125936.107099999 484861.906500002,125937.0601 484881.625500001,125940.576099999 484900.000500002,125947.870231801 484916.8,126074.2 484916.8,126074.2 484750.6,125935.34666449 484750.6))'

# Tweak yourself!
dbhost = 'host_name'
dbname = 'db_name'
port = '5432'
user = 'user_name'
password = 'secret'

schema = 'kerngis'
table = 'beheergebied'
where = 'objectid = 99'

# defining the object that holds the points, the masks and more
if __name__ == '__main__':
    leda = psycopg2.connect(
        host=dbhost,
        database=dbname,
        port=port,
        user=user,
        connect_timeout=5)

    results, _ = execute_query(leda,
                               f'SELECT ST_AsText(geom) geom FROM {schema}.{table} WHERE {where}' )

    # if no acces to test area:
    # tree = DetectorTree(kerngis_test_area)
    tree = DetectorTree(results[0]['geom'])

    # first clustering step
    tree.hdbscan_on_points(min_cluster_size=30, min_samples=10, xyz=False)
    tree.convex_hullify(points=tree.clustered_points)
    df_to_pg(tree.tree_df, schema='bomen', table_name='xy_bomen')

    # second cluster step
    tree.find_points_in_polygons(tree.tree_df)
    tree.kmean_cluster(tree.xy_grouped_points, round_val=2)
    tree.convex_hullify(tree.kmean_grouped_points, kmean_pols=True)
    df_to_pg(tree.tree_df, schema='bomen', table_name='km_bomen')

    # hacky colors/visualizing
    write_df = color_clusters(tree.kmean_grouped_points)
    dataframe_to_laz(write_df, 'point_output.laz')
