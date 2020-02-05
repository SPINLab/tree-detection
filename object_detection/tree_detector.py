import json
import traceback
import time

import numpy as np
import pandas as pd
import pdal
from geopandas import GeoDataFrame
from scipy.spatial.qhull import ConvexHull
from shapely import geometry
from shapely.geometry import Point
from shapely.wkt import loads
from sklearn.preprocessing import StandardScaler
from numpy.lib import recfunctions as rfn
from hdbscan import HDBSCAN


class Detector_tree:
    '''

    '''

    def __init__(self, box):
        self.box = box
        self.xmin, self.ymin, self.xmax, self.ymax = box
        self.wkt_box = geometry.box(self.xmin, self.ymin, self.xmax, self.ymax).wkt
        self.raw_points = self.ept_reader(self.wkt_box)
        print(f'Total amount of points read: {self.raw_points.shape[0]}')
        self.tree_df = GeoDataFrame({'clusterID': [],
                                     'geometry': []})

        # Masks
        self.groundmask = self.raw_points['Classification'] == 2
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

        f_pts = f_pts[np.logical_and(self.n_returnsmask, self.groundmask)]
        data = f_pts.drop(['X', 'Y', 'Z'], axis=1)
        scaler = StandardScaler()
        scaler.fit(data)
        normalized_pointcloud = pd.DataFrame(scaler.transform(data), columns=data.columns)
        normalized_pointcloud = f_pts[['X', 'Y', 'Z']].join(normalized_pointcloud)
        normalized_pointcloud['pid'] = normalized_pointcloud.index

        return normalized_pointcloud

    def create_writable_pointcloud(self, initial_columns=['X', 'Y', 'Z', 'Red', 'Green', 'Blue', 'Intensity', 'ReturnNumber'],
                                   added_column_name='', added_column_data=np.array([])):
        if not self.mask:
            print('I did not find a mask, created writable pointcloud from copy of input')
            self.mask = [True]*self.raw_points.shape[0]

        if initial_columns:
            self.out_pts = self.raw_points[initial_columns]
            if added_column_name:
                rfn.append_fields(
                    added_column_data,
                    'Classification',
                    self.normalized_pointcloud[self.mask][added_column_name]
                )

    def cluster_on_xy(self, min_cluster_size, min_samples):
        start = time.time()
        xy = self.raw_points[['X', 'Y']][np.logical_and(self.groundmask, self.n_returnsmask)]
        xy = np.array([xy['X'],
                       xy['Y']]).T
        xy_clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                               min_samples=min_samples)
        xy_clusterer.fit(xy)
        self.normalized_pointcloud['xy_clusterID'] = xy_clusterer.labels_
        end = time.time()
        print(f'clustering took {round(end - start, 2)} seconds')

    def kmean_cluster(self):
        # TODO points of interest are the non-masked points within a tree cluster
        xyz = [Point(coords) for coords in zip(self.raw_points['X'], self.raw_points['Y'], self.raw_points['Z'])]
        self.pre_process(self.raw_points[self.groundmask]['Red', 'Green', 'Blue', 'Intensity', 'ReturnNumber', 'NumberOfReturns'])
        for treeID in self.tree_df.clusterID:
            if treeID >= 0:
                geometry = self.tree_df.loc[treeID, 'geometry']



        for name, group in st_data.groupby('xy_clusterID'):
            if group.shape[0] >= 2000:
                value_clusterer = HDBSCAN(min_cluster_size=40, min_samples = 10)
                value_clusterer.fit(group.loc[:,['X','Y','Z']].T)
                labs = np.append(labs, value_clusterer.labels_)
            else:
                labs = np.append(labs, [1] * group.shape[0])

    def convex_hullify(self):
        for name, group in self.normalized_pointcloud.groupby('xy_clusterID'):
            coords = np.array([group.X, group.Y]).T
            polygon = ConvexHull(coords)
            wkt = 'POLYGON (('
            for id in polygon.vertices:
                x, y = polygon.points[id]
                wkt += f'{x} {y},'
            # close the polygon
            firstx, firsty = polygon.points[polygon.vertices[0]]
            wkt = wkt + f'{firstx} {firsty}))'
            print(wkt)
            self.tree_df.loc[len(self.tree_df)] = [int(name), loads(wkt)]



