import json
import traceback
import time

import numpy as np
import pandas as pd
import pdal
from geoalchemy2 import WKTElement, Geometry
from geopandas import GeoDataFrame, sjoin
from kneed import KneeLocator
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from scipy.spatial.qhull import ConvexHull
from shapely import geometry
from shapely.geometry import Point
from shapely.wkt import loads
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
from sqlalchemy import create_engine
from skimage.feature import peak_local_max
from object_detection.helper_functions import round_to_val


class Detector_tree:
    '''

    '''

    def __init__(self, box):
        self.box = box
        self.xmin, self.ymin, self.xmax, self.ymax = box
        self.wkt_box = geometry.box(self.xmin, self.ymin, self.xmax, self.ymax).wkt
        start = time.time()
        self.raw_points = self.ept_reader(self.wkt_box)
        end = time.time()
        print(f'reading from the ept took {round(end - start,2)}')
        print(f'Total amount of points read: {self.raw_points.shape[0]}')
        self.tree_df = GeoDataFrame({'xy_clusterID': [],
                                     'geometry': []})

        # Masks
        self.groundmask = self.raw_points['Classification'] != 2
        self.n_returnsmask = self.raw_points['NumberOfReturns'] >= 3

    def ept_reader(self, polygon_wkt: str) -> np.ndarray:
        """
            Parameters
            ----------
                Path to ept directory
            :param polygon_wkt : wkt
                WKT with clipping polygon

            Returns
            -------
            points : (Mx3) array
                The ept points
        """
        polygon = loads(polygon_wkt)
        bbox = polygon.bounds
        ept_location: str = 'http://ngi.geodan.nl/maquette/colorized-points/ahn3_nl/ept-subsets/ept.json'
        bounds = f"([{bbox[0]},{bbox[2]}],[{bbox[1]},{bbox[3]}])"

        pipeline_config = {
            "pipeline": [
                {
                    "type": "readers.ept",
                    "filename": ept_location,
                    "bounds": bounds
                },
                {
                    "type": "filters.crop",
                    "polygon": polygon_wkt
                },
                {
                    "type": "filters.smrf"
                }
            ]
        }

        try:
            pipeline = pdal.Pipeline(json.dumps(pipeline_config))
            pipeline.validate()  # check if our JSON and options were good
            pipeline.execute()
        except Exception as e:
            trace = traceback.format_exc()
            print("Unexpected error:", trace)
            print('Polygon:', polygon_wkt)
            print("Error:", e)
            raise

        arrays = pipeline.arrays
        points = arrays[0]
        return points

    def preprocess(self, points):
        f_pts = pd.DataFrame(points)

        f_pts['pid'] = f_pts.index
        columns_to_keep = [column for column in f_pts.columns if column not in ['pid', 'X', 'Y', 'Z']]
        scaler = StandardScaler()
        scaler.fit(f_pts[columns_to_keep])

        f_pts[columns_to_keep] = scaler.transform(f_pts[columns_to_keep])

        normalized_pointcloud = pd.DataFrame(f_pts,
                                             columns=f_pts.columns)
        return normalized_pointcloud

    def cluster_on_xy(self, min_cluster_size, min_samples):
        masked_points = self.raw_points[np.logical_and(self.groundmask, self.n_returnsmask)]
        start = time.time()
        xy = np.array([masked_points['X'], masked_points['Y']]).T

        xy_clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                               min_samples=min_samples)
        xy_clusterer.fit(xy)
        self.clustered_points = pd.DataFrame({'X': masked_points['X'],
                                              'Y': masked_points['Y'],
                                              'Z': masked_points['Z'],
                                              'Red': masked_points['Red'],
                                              'Green': masked_points['Green'],
                                              'Blue': masked_points['Blue'],
                                              'Classification': xy_clusterer.labels_})
        self.clustered_points = self.clustered_points[self.clustered_points.Classification >= 0]
        end = time.time()
        print(f'clustering on xy took {round(end - start, 2)} seconds')

    def convex_hullify(self, points):
        try:
            self.tree_df.drop(self.tree_df.index, inplace=True)
        except:
            pass

        for name, group in points.groupby('Classification'):
            if group.shape[0] <= 3:
                points.drop(points.groupby('Classification').get_group(name).index)
            else:
                coords = np.array([group.X, group.Y]).T
                polygon = ConvexHull(coords)

                # build wkt string
                wkt = 'POLYGON (('
                for group_id in polygon.vertices:
                    x, y = polygon.points[group_id]
                    wkt += f'{x} {y},'
                # close the polygon
                firstx, firsty = polygon.points[polygon.vertices[0]]
                wkt = wkt + f'{firstx} {firsty}))'

                # if there are less than 8 points per square meter; it's not a tree
                if (group.shape[0] / loads(wkt).area) <= 8:
                    points.drop(points.groupby('Classification').get_group(name).index)
                else:
                # write to df
                    self.tree_df.loc[len(self.tree_df)] = [int(name), loads(wkt)]

    def find_points_in_polygons(self):
        cluster_points = self.raw_points[self.groundmask]
        xy = [Point(coords) for coords in zip(cluster_points['X'], cluster_points['Y'], cluster_points['Z'])]

        cluster_data = self.preprocess(
            cluster_points[['X', 'Y', 'Z',
                            'Red', 'Green', 'Blue',
                            'Intensity', 'ReturnNumber', 'NumberOfReturns']]
        )

        points_df = GeoDataFrame(cluster_data, geometry=xy)
        grouped_points = sjoin(points_df, self.tree_df, how='left')
        grouped_points['X'] = grouped_points.geometry.apply(lambda p: p.x)
        grouped_points['Y'] = grouped_points.geometry.apply(lambda p: p.y)

        # TODO no idea where the nans are coming from
        # dirty hack, hope not to many points go missing
        print(f'removing {np.isnan(grouped_points.index_right).sum()} mystery nans <- merge_points_polygons')
        grouped_points = grouped_points[~np.isnan(grouped_points.index_right)]

        # remove noise
        print(f'Removed {np.array([grouped_points.xy_clusterID < 0]).sum()} noise points')
        grouped_points = grouped_points[grouped_points.xy_clusterID >= 0]
        self.xy_grouped_points = grouped_points.rename(columns={'index_right': 'polygon_id'})

    def kmean_cluster(self, xy_grouped_points):
        # TODO: see if it is possible to use initial clusterpoints
        labs = np.array([])
        n_ids = np.array([])
        self.kmean_grouped_points = xy_grouped_points.copy()

        for name, group in self.kmean_grouped_points.groupby('xy_clusterID'):
            tree_area = float(self.tree_df.loc[self.tree_df['xy_clusterID'] == int(name)].geometry.area)
            if name >= 0 and tree_area >= 2:
                cluster_data = np.array([group.X,
                                         group.Y,
                                         group.Z]).T

                n_clusters = self.find_n_clusters_peaks(cluster_data,
                                                        tree_area,
                                                        min_dist=2,
                                                        min_height=0,  # min(group.Y) + 1,
                                                        gridsize=1.5)

                print(f'polygon: {int(name)}  \t area:  {round(tree_area, 2)} \t Found {n_clusters} clusters')
                kmeans = KMeans(n_clusters=n_clusters).fit(cluster_data)
                labs = np.append(labs, kmeans.labels_)

            else:
                # TODO figure out a way to find a label not in the kmeans lables
                labs = np.append(labs, [1] * group.shape[0])
            n_ids = np.append(n_ids, group.pid)

        arr_inds = n_ids.argsort()
        sorted_labs = labs[arr_inds]
        self.kmean_grouped_points['value_clusterID'] = sorted_labs * 10
        combi_ids = ["".join(row) for row in
                     self.kmean_grouped_points[['value_clusterID', 'xy_clusterID']].values.astype(str)]
        self.kmean_grouped_points['Classification'] = pd.factorize(combi_ids)[0]

    def find_n_clusters(self, cluster_data, tree_area):
        # TODO there are between (area / 200) and (area / 40) clusters per polygon
        krange_min = max(1, round(tree_area / 200))
        krange_max = max(1, round(tree_area / 40))
        print(f'# points: {cluster_data.shape[0]} \t '
              f'Density: {round(cluster_data.shape[0] / tree_area, 2)} pts/m2')

        distortions = []
        kmean_range = range(krange_min, krange_max)
        if len(kmean_range) <= 1:
            return krange_min

        for k in kmean_range:
            kmeans = KMeans(n_clusters=k).fit(cluster_data)
            distortions.append(
                sum(np.min(cdist(cluster_data, kmeans.cluster_centers_, 'euclidean'), axis=1)) /
                cluster_data.shape[0])
        print(distortions)
        knee = KneeLocator(x=range(1, len(distortions) + 1),
                           y=distortions,
                           curve='convex',
                           direction='decreasing')

        # TODO: handle this better
        if knee.knee:
            return knee.knee
        else:
            return round(krange_max / krange_min)

    def find_n_clusters_peaks(self, cluster_data, tree_area, gridsize, min_dist, min_height):

        # round the data
        d_round = np.empty([cluster_data.shape[0], 5])

        d_round[:, 0] = cluster_data[:, 0]
        d_round[:, 1] = cluster_data[:, 1]
        d_round[:, 2] = cluster_data[:, 2]
        d_round[:, 3] = round_to_val(d_round[:, 0], gridsize)
        d_round[:, 4] = round_to_val(d_round[:, 1], gridsize)

        df = pd.DataFrame(d_round, columns=['x', 'y', 'z', 'x_round', 'y_round'])
        df_round = df[['x_round', 'y_round', 'z']]
        binned_data = df_round.groupby(['x_round', 'y_round'], as_index=False).count()

        minx, maxx = min(df.x), max(df.x)
        miny, maxy = min(df.y), max(df.y)

        # binned_data = np.loadtxt('your_binned_data.csv', skiprows=1, delimiter=',')
        x_bins = binned_data.x_round
        y_bins = binned_data.y_round
        z_vals = binned_data.z

        pts = np.array([x_bins, y_bins])
        pts = pts.T

        grid_x, grid_y = np.mgrid[minx:maxx:gridsize, miny:maxy:gridsize]

        # interpolate onto grid
        data_grid = griddata(pts, z_vals, (grid_x, grid_y), method='cubic')
        data_grid = np.nan_to_num(data_grid, 0)
        coordinates = peak_local_max(data_grid, min_distance=min_dist, threshold_abs= min_height )
        n_cluster = coordinates.shape[0]

        return max(1, n_cluster)

    def df_to_PG(self,
                 input_gdf,
                 schema,
                 table_name,
                 database='VU',
                 port='5432',
                 host='leda.geodan.nl',
                 username='arnot',
                 password=''):

        geodataframe = input_gdf.copy()
        engine = create_engine(f'postgresql://{username}@{host}:{port}/{database}')
        geodataframe['geom'] = geodataframe['geometry'].apply(lambda x: WKTElement(x.wkt, srid=28992))
        geodataframe.drop('geometry', 1, inplace=True)
        print('warning! For now everything in the database is replaced!!!')

        geodataframe.to_sql(table_name,
                            engine,
                            if_exists='replace',
                            index=False,
                            schema=schema,
                            dtype={'geom': Geometry('Polygon', srid=28992)})
