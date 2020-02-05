import json
import traceback

import numpy as np
import pandas as pd
import pdal
from shapely import geometry
from shapely.wkt import loads
from sklearn.preprocessing import StandardScaler
from numpy.lib import recfunctions as rfn


class Detector_tree:
    '''

    '''

    def __init__(self, box):
        self.box = box
        self.xmin, self.ymin, self.xmax, self.ymax = box
        self.wkt_box = geometry.box(self.xmin, self.ymin, self.xmax, self.ymax).wkt
        self.raw_points = self.ept_reader(self.wkt_box)

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
                    "type": "filters.range",
                    "limits": "NumberOfReturns[3:]"
                },
                {
                    "type": "filters.smrf"
                },
                {
                    "type": "filters.range",
                    "limits": "Classification[:1]"
                },

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

    def preprocess(self):
        f_pts = pd.DataFrame(self.raw_points[['X', 'Y', 'Z', 'Red', 'Green', 'Blue', 'Intensity', 'ReturnNumber']])
        data = f_pts.drop(['X', 'Y', 'Z'], axis=1)
        scaler = StandardScaler()
        scaler.fit(data)
        self.normalized_pointcloud = pd.DataFrame(scaler.transform(data), columns=data.columns)
        self.st_dnormalized_pointcloudata = f_pts[['X', 'Y', 'Z']].join(self.normalized_pointcloud)
        self.normalized_pointcloud['pid'] = self.normalized_pointcloud.index

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
