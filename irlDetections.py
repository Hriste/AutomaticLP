'''
Christina Paolicelli
5/13/20

Detections on IRL images
'''
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib
import sys
from matplotlib import pyplot as plt
sys.path.append("models/research/object_detection")
from utils import label_map_util
from utils import visualization_utils as vis_util


NUM_CLASSES = 32
MODEL_NAME = 'inference_graph'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = "inference_graph/frozen_inference_graph.pb"
PATH_TO_LABELS = "FromScratch/data/label_map.pbtxt"

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detectedd
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

threshold = 0.5

matplotlib.use('tkAgg')

imageDirectory = input("Enter path to directory of images to evaluate: ")
for filename in os.listdir(imageDirectory):
    pathToImage = os.path.join(CWD_PATH, imageDirectory, filename)
    image = cv2.imread(pathToImage)
    image_expanded = np.expand_dims(image, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=float(threshold),
        groundtruth_box_visualization_color = 'red')
    plt.imshow(image)
    plt.show()
