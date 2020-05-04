'''
Christina Paolicelli
March 15th 2020

Converter between my CSV output and TF Record for Tensorflow Training and Evaluation
'''

import hashlib
import csv
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import os
import tensorflow as tf

CLASS_NAMES = ['','1','2','3','4','5','6','7','8','9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z','0']
PATH_TO_LABELS = './data/label_map.pbtxt'

def createTFRecord(row, imgPath):
    height = 256
    width = 512
    image_format = 'jpeg'.encode('utf-8')

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    # We know there are 7 characters in the LP
    filename = row[0].encode('utf-8')
    # 1st Character
    classes_text.append(CLASS_NAMES[int(row[1])].encode('utf-8'))
    classes.append(int(row[1]))
    xmins.append(float(row[2])/width)
    ymins.append(float(row[3])/height)
    xmaxs.append(float(row[4])/width)
    ymaxs.append(float(row[5])/height)
    # 2nd Character
    classes_text.append(CLASS_NAMES[int(row[6])].encode('utf-8'))
    classes.append(int(row[6]))
    xmins.append(float(row[7])/width)
    ymins.append(float(row[8])/height)
    xmaxs.append(float(row[9])/width)
    ymaxs.append(float(row[10])/height)
    # 3rd Character
    classes_text.append(CLASS_NAMES[int(row[11])].encode('utf-8'))
    classes.append(int(row[11]))
    xmins.append(float(row[12])/width)
    ymins.append(float(row[13])/height)
    xmaxs.append(float(row[14])/width)
    ymaxs.append(float(row[15])/height)
    # 4th Character
    classes_text.append(CLASS_NAMES[int(row[16])].encode('utf-8'))
    classes.append(int(row[16]))
    xmins.append(float(row[17])/width)
    ymins.append(float(row[18])/height)
    xmaxs.append(float(row[19])/width)
    ymaxs.append(float(row[20])/height)
    # 5th Character
    classes_text.append(CLASS_NAMES[int(row[21])].encode('utf-8'))
    classes.append(int(row[21]))
    xmins.append(float(row[22])/width)
    ymins.append(float(row[23])/height)
    xmaxs.append(float(row[24])/width)
    ymaxs.append(float(row[25])/height)
    # 6th Character
    classes_text.append(CLASS_NAMES[int(row[26])].encode('utf-8'))
    classes.append(int(row[26]))
    xmins.append(float(row[27])/width)
    ymins.append(float(row[28])/height)
    xmaxs.append(float(row[29])/width)
    ymaxs.append(float(row[30])/height)
    # 7th Character
    classes_text.append(CLASS_NAMES[int(row[31])].encode('utf-8'))
    classes.append(int(row[31]))
    xmins.append(float(row[32])/width)
    ymins.append(float(row[33])/height)
    xmaxs.append(float(row[34])/width)
    ymaxs.append(float(row[35])/height)



    img_path = os.path.join(imgPath, row[0])
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()

    key = hashlib.sha256(encoded_jpg).hexdigest()

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/key/sha256':dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded':dataset_util.bytes_feature(encoded_jpg),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      }))
    return tf_example

def _parse_(serialized_example):
    feature = {'image_raw':tf.FixedLenFeature([],tf.string),
                'label':tf.FixedLenFeature([],tf.int64)}
    example = tf.parse_single_example(serialized_example,feature)
    image = tf.decode_raw(example['image_raw'],tf.int64) #remember to parse in int64. float will raise error
    label = tf.cast(example['label'],tf.int32)
    return (dict({'image':image}),label)

def tfrecord_train_input_fn(batch_size=32):
    tfrecord_dataset = tf.data.TFRecordDataset("TFRecord.tfrecord")
    tfrecord_dataset = tfrecord_dataset.map(lambda   x:_parse_(x)).shuffle(True).batch(batch_size)
    tfrecord_iterator = tfrecord_dataset.make_one_shot_iterator()

    return tfrecord_iterator.get_next()
