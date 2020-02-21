import json
import traceback
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import pdal
from geopandas import GeoDataFrame, sjoin
from scipy.spatial.qhull import ConvexHull
from shapely import geometry
from shapely.geometry import Point
from shapely.wkt import loads
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
from object_detection.helper_functions import find_n_clusters_peaks, df_to_pg, preprocess, dataframe_to_laz


# noinspection PyAttributeOutsideInit
class DetectorTree:
    '''

    '''

    def __init__(self, box):
        self.box = box
        self.xmin, self.ymin, self.xmax, self.ymax = box
        self.geom_box = geometry.box(self.xmin, self.ymin, self.xmax, self.ymax)
        self.wkt_box = self.geom_box.wkt
        start = time.time()
        self.raw_points = self.ept_reader(self.wkt_box)
        end = time.time()
        print(f'reading from the ept took {round(end - start, 2)}')
        print(f'Total amount of points read: {self.raw_points.shape[0]}')
        self.tree_df = GeoDataFrame({'xy_clusterID': [],
                                     'geometry': []})

        # Masks
        self.ground_mask = self.raw_points['Classification'] == 2
        self.n_returns_mask = self.raw_points['NumberOfReturns'] < 2
        self.coplanar_mask = self.raw_points['Coplanar'] == 1
        self.linearity_mask = np.logical_or(self.raw_points['Planarity'] > 0.4,
                                            self.raw_points['Linearity'] > 0.4)
        # self.planarity_mask = self.raw_points['Planarity'] > 0.4
        self.radialdensity_mask = self.raw_points['RadialDensity'] < 0

        self.tree_coords = pd.DataFrame(data={'X': [],
                                              'Y': []})

        # masks = np.vstack([self.ground_mask, self.n_returns_mask, self.linearity_mask, self.coplanar_mask])  #
        masks = np.vstack([self.ground_mask, self.n_returns_mask, self.radialdensity_mask, self.linearity_mask])
        self.masks = np.sum(masks, axis=0) == 0

        df_to_pg(GeoDataFrame(
            data={'what': 'boundingbox',
                  'geometry': self.geom_box},
            index=[0]),
            schema='bomen',
            table_name='bbox')

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
                    "type": "filters.approximatecoplanar",
                    "knn": 8,
                    "thresh1": 25,
                    "thresh2": 6
                },
                {
                    "type": "filters.covariancefeatures",
                    "knn": 8,
                    "threads": 6,
                    "feature_set": "Dimensionality"
                },
                {
                    "type": "filters.outlier"
                },
                {
                    "type": "filters.smrf",
                    "returns": "last,only",
                    "slope": 0.2,
                    "window": 18,
                    "threshold": 0.45,
                    "scalar": 1.2,
                    "ignore": "Classification[7:7]",
                },
                {
                    "type": "filters.hag"
                },
                {
                    "type": "filters.radialdensity",
                    "radius": 2.0
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

    def cluster_on_xy(self, min_cluster_size, min_samples):

        masked_points = self.raw_points[self.masks]

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
                                              'HAG': masked_points['HeightAboveGround'],
                                              'Linearity': masked_points['Linearity'],
                                              'Planarity': masked_points['Planarity'],
                                              'Scattering': masked_points['Scattering'],
                                              'Verticality': masked_points['Verticality'],
                                              'Classification': xy_clusterer.labels_})
        # remove "noise" points
        self.clustered_points = self.clustered_points[self.clustered_points.Classification >= 0]
        end = time.time()
        print(f'found {np.unique(len(np.unique(self.clustered_points.Classification)))[0]} xy_clusters')
        print(f'clustering on xy took {round(end - start, 2)} seconds')

    def convex_hullify(self, points):
        try:
            self.tree_df.drop(self.tree_df.index, inplace=True)
        except:
            pass

        for name, group in points.groupby('Classification'):
            if group.shape[0] <= 3:
                # remove polygons that contain too little points to hullify around
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
                if (group.shape[0] / loads(wkt).area) <= 5:
                    points.drop(points.groupby('Classification').get_group(name).index)
                else:
                    # write to df
                    self.tree_df.loc[len(self.tree_df)] = [int(name), loads(wkt)]

    def find_points_in_polygons(self, polygon_df):
        cluster_points = self.raw_points[self.ground_mask.__invert__()]
        xy = [Point(coords) for coords in zip(cluster_points['X'], cluster_points['Y'], cluster_points['Z'])]

        cluster_data = preprocess(
            cluster_points[['X', 'Y', 'Z',
                            'Red', 'Green', 'Blue',
                            'Intensity', 'ReturnNumber', 'NumberOfReturns',
                            'HeightAboveGround']])

        points_df = GeoDataFrame(cluster_data, geometry=xy)
        grouped_points = sjoin(points_df, polygon_df, how='left')
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

    def kmean_cluster(self, xy_grouped_points, min_dist, min_height, gridsize):
        # TODO: see if it is possible to use initial clusterpoints
        labs = np.array([])
        n_ids = np.array([])
        self.kmean_grouped_points = xy_grouped_points.copy()

        for name, group in self.kmean_grouped_points.groupby('xy_clusterID'):
            tree_area = float(self.tree_df.loc[self.tree_df['xy_clusterID'] == int(name)].geometry.area)
            if name >= 0 and tree_area >= 2 and group.shape[0] >= 10:
                kmeans_labels = self.kmean_cluster_group(group, min_dist, min_height, gridsize)
                labs = np.append(labs, kmeans_labels)
                print(f'polygon: {int(name)}  \t area:  {round(tree_area, 2)} \t Found {len(np.unique(kmeans_labels))} clusters')

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

    def kmean_cluster_group(self, group, min_dist, min_height, gridsize):
        cluster_data = np.array([group.X,
                                 group.Y,
                                 group.Z]).T

        n_clusters, coordinates = find_n_clusters_peaks(cluster_data,
                                                        min_dist=min_dist,
                                                        # is rounded to a multiple of the gridsize
                                                        min_height=min_height,  # min(group.Y) + 1,
                                                        grid_size=gridsize)

        kmeans = KMeans(n_clusters=n_clusters).fit(cluster_data)

        return kmeans.labels_

    # def find_stems(self, points, grid_size, min_dist):
    #
    #     # TODO read dataframe with pdal and detect lines
    #     # stems are lines
    #     for name, group in points.groupby('xy_clusterID'):
    #         group = group[group.HeightAboveGround <= 2]
    #         cluster_data = np.array([group.X,
    #                                  group.Y,
    #                                  group.Z * -1]).T
    #
    #         stems, coordinates = find_n_clusters_peaks(cluster_data, grid_size, min_dist, 0)  # 0 = min_height
    #
    #     coordinates = np.array(coordinates).T
    #     if coordinates.shape[0] > 0:
    #         tmp = pd.DataFrame({'X': coordinates[0],
    #                             'Y': coordinates[1]})
    #
    #         print(tmp)
    #         tree_coords = self.tree_coords.copy()
    #         tree_coords = tree_coords.append(tmp, ignore_index=True)
    #         self.tree_coords = GeoDataFrame(
    #             tree_coords, geometry=gpd.points_from_xy(tree_coords.X, tree_coords.Y))

    def find_stems(self, points):
        pipeline_config = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": "tmp.laz"
                },
                {
                    "type": "filters.smrf"
                },
                {
                    "type": "filters.hag"
                },
                {
                    "type": "filters.outlier",
                    "method": "statistical",
                    "mean_k": 12,
                    "multiplier": 2.2
                },
                {
                    "type": "filters.range",
                    "limits": "HeightAboveGround[0.5:), Classification![7:7]"
                }

            ]
        }

        for name, group in points.groupby('xy_clusterID'):
            # group = group[group.HeightAboveGround <= 2]
            group = group[['X', 'Y', 'Z', 'pid']]
            dataframe_to_laz(group, 'tmp.laz')

            try:
                p = pdal.Pipeline(json=json.dumps(pipeline_config))
                p.validate()  # check if our JSON and options were good
                p.execute()

            except Exception as e:
                trace = traceback.format_exc()
                print("Unexpected error:", trace)
                raise

            arrays = p.arrays
            points = arrays[0]
            return points
