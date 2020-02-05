from object_detection.tree_detector import Detector_tree

tree = Detector_tree((122619.6, 490344.5, 122668.2, 490373.4))
tree.preprocess()
print(tree.normalized_pointcloud)