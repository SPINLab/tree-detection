import numpy as np
import pdal
from numpy.lib import recfunctions as rfn
import pandas as pd


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

def dataframe_to_laz(dataframe, laz_fn):
    result = dataframe.to_records()
    write_to_laz(result, laz_fn)

def round_to_val(a, round_val):
    return np.round(np.array(a, dtype=float) / round_val) * round_val