import os
from random import sample

import numpy as np
import pdal
from geoalchemy2 import WKTElement, Geometry
import pandas as pd
from skimage.feature import peak_local_max
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler


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

    result = dataframe.to_records()
    write_to_laz(result, laz_fn)


def round_to_val(a, round_val):
    return np.round(np.array(a, dtype=float) / round_val) * round_val


def find_n_clusters_peaks(cluster_data, grid_size, min_dist, min_height):
    points = pd.DataFrame(data=cluster_data,
                          columns=['X', 'Y', 'Z'])

    img, minx, miny = interpolate_df(points, grid_size)

    indices = peak_local_max(img, min_distance=min_dist, threshold_abs=min_height)
    n_cluster = indices.shape[0]

    # TODO return coordinates
    mins = [[minx, miny]] * indices.shape[0]
    mapped = map(add_vectors, zip(indices, mins))
    coordinates = [coord for coord in mapped]

    return max(1, n_cluster), coordinates


def interpolate_df(xyz_points, grid_size):
    xyz_points['x_round'] = round_to_val(xyz_points.X, grid_size)
    xyz_points['y_round'] = round_to_val(xyz_points.Y, grid_size)

    binned_data = xyz_points.groupby(['x_round', 'y_round'], as_index=False).max()

    minx = min(binned_data.x_round)
    miny = min(binned_data.y_round)

    x_arr = binned_data.x_round - min(binned_data.x_round)
    y_arr = binned_data.y_round - min(binned_data.y_round)

    img = np.zeros([int(max(y_arr)) + 1, int(max(x_arr)) + 1])

    img[y_arr.astype(np.int).values, x_arr.astype(np.int).values] = binned_data.Z

    return img, minx, miny


def df_to_pg(input_gdf,
             schema,
             table_name,
             database='VU',
             port='5432',
             host='leda.geodan.nl',
             username='arnot',
             password=''):
    geo_dataframe = input_gdf.copy()
    engine = create_engine(f'postgresql://{username}@{host}:{port}/{database}')
    geo_dataframe['geom'] = geo_dataframe['geometry'].apply(lambda x: WKTElement(x.wkt, srid=28992))
    geo_dataframe.drop('geometry', 1, inplace=True)
    print('warning! For now everything in the database is replaced!!!')

    geo_dataframe.to_sql(table_name,
                         engine,
                         if_exists='replace',
                         index=False,
                         schema=schema,
                         dtype={'geom': Geometry(input_gdf.geometry.geom_type[0], srid=28992)})


def preprocess(points):
    f_pts = pd.DataFrame(points)
    f_pts['pid'] = f_pts.index
    return f_pts

    columns_to_keep = [column
                       for column in f_pts.columns
                       if column not in ['pid',
                                         'X', 'Y', 'Z',
                                         'Red', 'Green', 'Blue',
                                         'Intensity', 'ReturnNumber', 'NumberOfReturns',
                                         'HeightAboveGround'
                                         ]]
    scaler = StandardScaler()
    scaler.fit(f_pts[columns_to_keep])
    f_pts[columns_to_keep] = scaler.transform(f_pts[columns_to_keep])
    normalized_pointcloud = pd.DataFrame(data=f_pts,
                                         columns=f_pts.columns)
    return normalized_pointcloud


def add_vectors(vec):
    a, b = vec
    return [a[0] + b[0], a[1] + b[1]]


def get_colors(n):
    cols = [[0, 0, 0], [1, 0, 103], [213, 255, 0], [255, 0, 86], [158, 0, 142],
            [14, 76, 161], [255, 229, 2], [0, 95, 57], [0, 255, 0], [149, 0, 58], [255, 147, 126], [164, 36, 0],
            [0, 21, 68], [145, 208, 203], [98, 14, 0], [107, 104, 130], [0, 0, 255], [0, 125, 181], [106, 130, 108],
            [0, 174, 126], [194, 140, 159], [190, 153, 112], [0, 143, 156], [95, 173, 78], [255, 0, 0], [255, 0, 246],
            [255, 2, 157], [104, 61, 59], [255, 116, 163], [150, 138, 232], [152, 255, 82], [167, 87, 64],
            [1, 255, 254], [255, 238, 232], [254, 137, 0], [189, 198, 255], [1, 208, 255], [187, 136, 0],
            [117, 68, 177], [165, 255, 210], [255, 166, 254], [119, 77, 0], [122, 71, 130], [38, 52, 0], [0, 71, 84],
            [67, 0, 44], [181, 0, 255], [255, 177, 103], [255, 219, 102], [144, 251, 146], [126, 45, 210],
            [189, 211, 147], [229, 111, 254], [222, 255, 116], [0, 255, 120], [0, 155, 255], [0, 100, 1], [0, 118, 255],
            [133, 169, 0], [0, 185, 23], [120, 130, 49], [0, 255, 198], [255, 110, 65], [232, 94, 190]]
    return sample(cols, n)
