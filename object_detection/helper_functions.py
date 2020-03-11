import json
import os
import traceback
from random import sample

import numpy as np
import pdal
from geoalchemy2 import WKTElement, Geometry
import pandas as pd
from shapely.wkt import loads
from skimage.feature import peak_local_max
from sqlalchemy import create_engine
from retry import retry


@retry(tries=5, delay=1, backoff=3, max_delay=30)
def ept_reader(polygon_wkt: str) -> np.ndarray:
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
    ept_location: str = 'https://beta.geodan.nl/maquette/colorized-points/ahn3_nl/ept-subsets/ept.json'
    bounds = f"([{bbox[0]},{bbox[2]}],[{bbox[1]},{bbox[3]}])"

    #:TODO poission sampling (0.25)
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
                # Actually filters the points that are not 'unclassified' by AHN
                "type": "filters.range",
                "limits": "Classification[1:1]"
            },
            {
                # ground filter
                "type": "filters.smrf",
                "scalar": 1.2,
                "slope": 0.2,
                "threshold": 0.45,
                "window": 16.0
            },
            {
                "type": "filters.hag"
            },
            {
                "type": "filters.ferry",
                "dimensions": "HeightAboveGround=HAG"
            },
            {
                # :TODO do i use this?
                "type": "filters.approximatecoplanar",
                "knn": 8,
                "thresh1": 25,
                "thresh2": 6
            },
            {
                # :TODO can this go?
                "type": "filters.normal",
                "knn": 8
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


def write_to_laz(structured_array, path):
    '''
    writes a structured array to a .laz file
    in:
        point_cloud [structured np array]:
            The output pointcloud; needs attributes x, y and z.
            When createing a pointcloud from scratch, pay attention to
            the data types of the specific attributes, this is a pain in the ass.
            Easier to add one new collumn to an existing (filtered) pointcloud.
        path [string]:
            Path to a laz file.
    out:
        None
    '''
    WRITE_PIPELINE = """
    {{
        "pipeline": [
            {{
                "type": "writers.las",
                "filename": "{path}",
                "extra_dims": "all"
            }}
        ]
    }}
    """
    pipeline = pdal.Pipeline(
        WRITE_PIPELINE.format(path=path),
        arrays=[structured_array]
    )

    pipeline.validate()
    pipeline.execute()


def dataframe_to_laz(dataframe, laz_fn, overwrite=True):
    if os.path.exists(laz_fn) and overwrite:
        os.remove(laz_fn)
        print(f'removed {laz_fn}')

    result = dataframe.to_records()
    write_to_laz(result, laz_fn)


def round_to_val(a, round_val):
    return np.round(np.array(a, dtype=float) / round_val) * round_val


def find_n_clusters_peaks(cluster_data, round_val, min_dist):
    # :TODO Exaggerate height? Z*Z?
    img, minx, miny = interpolate_df(cluster_data, round_val)

    indices = peak_local_max(img, min_distance=min_dist)

    n_cluster = indices.shape[0]

    # TODO return coordinates
    mins = [[minx, miny, 0]] * indices.shape[0]
    z = [img[i[0], i[1]] for i in indices]
    round_val_for_map = [round_val] * n_cluster
    mapped = map(add_vectors, zip(indices, mins, z, round_val_for_map))
    coordinates = [coord for coord in mapped]

    return max(1, n_cluster), coordinates


def add_vectors(vec):
    coords, mins, z, round_val = vec
    y, x = coords
    minx, miny, minz = mins
    return [minx + (x * round_val), miny + (y * round_val), z]


def interpolate_df(xyz_points, round_val):
    xyz_points = xyz_points.T
    xyz_points = pd.DataFrame({'X': xyz_points[0],
                               'Y': xyz_points[1],
                               'Z': xyz_points[2]})

    xyz_points['x_round'] = round_to_val(xyz_points.X, round_val)
    xyz_points['y_round'] = round_to_val(xyz_points.Y, round_val)

    binned_data = xyz_points.groupby(['x_round', 'y_round'], as_index=False).max()

    minx = min(binned_data.x_round)
    miny = min(binned_data.y_round)

    x_arr = binned_data.x_round - min(binned_data.x_round)
    y_arr = binned_data.y_round - min(binned_data.y_round)

    img_size_x = int(round(max(x_arr), 1))
    img_size_y = int(round(max(y_arr), 1))

    img = np.zeros([img_size_x + 200, img_size_y + 200])

    img[round_to_val(y_arr / round_val, 1).astype(np.int),
        round_to_val(x_arr / round_val, 1).astype(np.int)] = binned_data.Z

    return img, minx, miny


def df_to_pg(input_gdf,
             schema,
             table_name,
             database='VU',
             port='5432',
             host='leda.geodan.nl',
             username='arnot',
             password=''):

    geo_dataframe = input_gdf.copy().reset_index()
    geom_type = geo_dataframe.geometry.geom_type[0]
    engine = create_engine(f'postgresql://{username}@{host}:{port}/{database}')
    geo_dataframe['geom'] = geo_dataframe['geometry'].apply(lambda x: WKTElement(x.wkt, srid=28992))
    geo_dataframe.drop('geometry', 1, inplace=True)
    print(f'warning! For now everything in {table_name} is replaced!!!')

    geo_dataframe.to_sql(table_name,
                         engine,
                         if_exists='replace',
                         index=False,
                         schema=schema,
                         dtype={'geom': Geometry(geom_type, srid=28992)})


def former_preprocess_now_add_pid(points):
    f_pts = pd.DataFrame(points)
    f_pts['pid'] = f_pts.index
    return f_pts


def get_colors(n):
    cols = 10 * [[0, 0, 0], [1, 0, 103], [213, 255, 0], [255, 0, 86], [158, 0, 142],
                 [14, 76, 161], [255, 229, 2], [0, 95, 57], [0, 255, 0], [149, 0, 58], [255, 147, 126], [164, 36, 0],
                 [0, 21, 68], [145, 208, 203], [98, 14, 0], [107, 104, 130], [0, 0, 255], [0, 125, 181],
                 [106, 130, 108], [0, 174, 126], [194, 140, 159], [190, 153, 112], [0, 143, 156], [95, 173, 78],
                 [255, 0, 0],
                 [255, 0, 246], [255, 2, 157], [104, 61, 59], [255, 116, 163], [150, 138, 232], [152, 255, 82],
                 [167, 87, 64],
                 [1, 255, 254], [255, 238, 232], [254, 137, 0], [189, 198, 255], [1, 208, 255], [187, 136, 0],
                 [117, 68, 177], [165, 255, 210], [255, 166, 254], [119, 77, 0], [122, 71, 130], [38, 52, 0],
                 [0, 71, 84], [67, 0, 44], [181, 0, 255], [255, 177, 103], [255, 219, 102], [144, 251, 146],
                 [126, 45, 210],
                 [189, 211, 147], [229, 111, 254], [222, 255, 116], [0, 255, 120], [0, 155, 255], [0, 100, 1],
                 [0, 118, 255], [133, 169, 0], [0, 185, 23], [120, 130, 49], [0, 255, 198], [255, 110, 65],
                 [232, 94, 190]]

    return sample(cols, n)