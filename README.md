# Tree Detection

This repository contains the methodology for detecting the trees in a project carried out for Rijkswaterstaat by the VU.

## Set up

`conda env create -f environment.yml --name <name>`

`activate <name>`

## How to use?
run: `python3 main.py`

## The steps for detecting trees are the following

1. Read points from the EPT
    1. PDAL filters for
    1. filter based on return number, classification and ground
    1. read points into class
1. HDBSCAN on XY
    1. assing cluster id
1. polygonize clusters with convex hull
1. get points in polygon (for better control over which points to filter in the next clustering step)
1. kmeans cluster with low number of iterations
    1. assing cluster id 
1. polygonize clusters based on unique combination of the two cluster id's


The code depends on a input dataset developed by [Geodan BV](https://www.geodan.nl/). The input dataset is an 
[entwined point tile](https://entwine.io/entwine-point-tile.html) (EPT) and is replacable. 
If you wish to use the Geodan EPT, please contact the owner of the repository.

