import json
import traceback
import time

import numpy as np
import pandas as pd
import pdal
from geopandas import GeoDataFrame, sjoin
from scipy.spatial.qhull import ConvexHull
from shapely import geometry
from shapely.geometry import Point
from shapely.wkt import loads
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
        self.tree_df = GeoDataFrame({'clusterID': [],
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
        # f_pts = f_pts[np.logical_and(self.n_returnsmask, self.groundmask)]
        data = f_pts.drop(['X', 'Y', 'Z'], axis=1)
        scaler = StandardScaler()
        scaler.fit(data)
        normalized_pointcloud = pd.DataFrame(scaler.transform(data), columns=data.columns)
        normalized_pointcloud['pid'] = normalized_pointcloud.index

        return normalized_pointcloud

    def cluster_on_xy(self, min_cluster_size, min_samples):
        start = time.time()
        xyz = self.raw_points[['X',
                               'Y',
                               'Z']][np.logical_and(self.groundmask, self.n_returnsmask)]
        xy = np.array([xyz['X'],
                       xyz['Y']]).T
        xy_clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                               min_samples=min_samples)
        xy_clusterer.fit(xy)
        self.clustered_points = pd.DataFrame({'X': xyz['X'],
                                              'Y': xyz['Y'],
                                              'Z': xyz['Z'],
                                              'Red': self.raw_points['Red'][np.logical_and(self.groundmask,
                                                                                           self.n_returnsmask)],
                                              'Green': self.raw_points['Green'][np.logical_and(self.groundmask,
                                                                                               self.n_returnsmask)],
                                              'Blue': self.raw_points['Blue'][np.logical_and(self.groundmask,
                                                                                             self.n_returnsmask)],
                                              'Classification': xy_clusterer.labels_})

        end = time.time()
        print(f'clustering took {round(end - start, 2)} seconds')

    def kmean_cluster(self, wkt_polygon):
        # TODO points of interest are the non-masked points within a tree cluster
        cluster_points = self.raw_points[self.groundmask]
        xyz = [Point(coords) for coords in zip(cluster_points['X'], cluster_points['Y'], cluster_points['Z'])]
        cluster_data = self.preprocess(cluster_points[['X', 'Y', 'Z', 'Red', 'Green', 'Blue', 'Intensity', 'ReturnNumber', 'NumberOfReturns']])
        gdf = GeoDataFrame(cluster_data, geometry=xyz)

        pointInPolys = sjoin(gdf, loads(wkt_polygon), how='left')

        return gdf

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
            print(wkt)
            self.tree_df.loc[len(self.tree_df)] = [int(name), loads(wkt)]



