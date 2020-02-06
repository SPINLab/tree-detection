import json
import traceback
import time

import numpy as np
import pandas as pd
import pdal
from geopandas import GeoDataFrame, sjoin
from kneed import KneeLocator
from scipy.spatial.qhull import ConvexHull
from shapely import geometry
from shapely.geometry import Point
from shapely.wkt import loads
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN


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
        scaler = StandardScaler().fit(f_pts[columns_to_keep])

        f_pts[columns_to_keep] = scaler.transform(f_pts[columns_to_keep])

        normalized_pointcloud = pd.DataFrame(f_pts,
                                             columns=f_pts.columns)

        # normalized_pointcloud = pd.merge(left=normalized_pointcloud,
        #                                  right=f_pts[['Z', 'pid']],
        #                                  left_on='pid',
        #                                  right_on='pid',
        #                                  how='left')

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

        end = time.time()
        print(f'clustering took {round(end - start, 2)} seconds')

    def convex_hullify(self, points):
        for name, group in points.groupby('Classification'):
            coords = np.array([group.X, group.Y]).T
            polygon = ConvexHull(coords)
            wkt = 'POLYGON (('
            for id in polygon.vertices:
                x, y = polygon.points[id]
                wkt += f'{x} {y},'
            # close the polygon
            firstx, firsty = polygon.points[polygon.vertices[0]]
            wkt = wkt + f'{firstx} {firsty}))'
            self.tree_df.loc[len(self.tree_df)] = [int(name), loads(wkt)]

    def merge_points_polygons(self):
        cluster_points = self.raw_points[self.groundmask]
        xy = [Point(coords) for coords in zip(cluster_points['X'], cluster_points['Y'], cluster_points['Z'])]

        cluster_data = self.preprocess(
            cluster_points[['X', 'Y', 'Z', 'Red', 'Green', 'Blue', 'Intensity', 'ReturnNumber', 'NumberOfReturns']]
        )

        points_df = GeoDataFrame(cluster_data, geometry=xy)
        grouped_points = sjoin(points_df, self.tree_df, how='left')
        grouped_points['X'] = grouped_points.geometry.apply(lambda p: p.x)
        grouped_points['Y'] = grouped_points.geometry.apply(lambda p: p.y)

        # TODO no idea where the nans are coming from
        # dirty hack, hope not to many points go missing
        print(f'removing {np.isnan(grouped_points.index_right).sum()} mystery nans <- merge_points_polygons')
        grouped_points = grouped_points[~np.isnan(grouped_points.index_right)]
        # grouped_points['Z'] = grouped_points.apply(lambda p: p.Z)

        self.grouped_points = grouped_points.rename(columns={'index_right': 'polygon_id'})

    def kmean_cluster(self):

        labs = np.array([])
        n_ids = np.array([])

        for name, group in self.grouped_points.groupby('xy_clusterID'):
            print(name)
            # TODO there are between area / 200 and area / 20 clusters per polygon
            krange = self.tree_df[name, 'geometry'].area
            if group.shape[0] >= 10000:
                print(f'there are {group.shape[0]} points in group {name}')
                cluster_data = np.array([group.X,
                                         group.Y,
                                         group.Z]).T

                n_clusters = self.find_n_clusters(cluster_data, krange)

                kmeans = KMeans(n_clusters=n_clusters).fit(cluster_data)
                labs = np.append(labs, kmeans.labels_)
            else:
                # TODO figure out a way to find a label not in the kmeans lables
                labs = np.append(labs, [1] * group.shape[0])
            n_ids = np.append(n_ids, group.pid)

        arr_inds = n_ids.argsort()
        sorted_labs = labs[arr_inds]
        self.grouped_points['value_clusterID'] = sorted_labs
        combi_ids = ["".join(row) for row in
                     self.grouped_points[['value_clusterID', 'xy_clusterID']].values.astype(str)]
        self.grouped_points['Classification'] = pd.factorize(combi_ids)[0]

    def find_n_clusters(self, cluster_data, krange):
        sum_squared_dist = []
        for k in krange:
            kmeans = KMeans(n_clusters=k).fit(cluster_data)
            sum_squared_dist.append(kmeans.inertia_)
        knee = KneeLocator(x=range(1, len(sum_squared_dist) + 1),
                           y=sum_squared_dist,
                           curve='convex',
                           direction='decreasing')
        return knee.knee





