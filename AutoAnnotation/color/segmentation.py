import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image as PIL_image


import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

#for segmentation
#://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb
#for image segmentation
#https://tech.ebayinc.com/research/modanet-a-large-scale-street-fashion-dataset-with-polygon-annotations/


########
'''
for transfer learning see this
https://www.freecodecamp.org/news/how-to-use-deeplab-in-tensorflow-for-object-segmentation-using-deep-learning-a5777290ab6b/
'''



class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path, labels):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')
    self.sess = tf.Session(graph=self.graph)
    self.LABEL_NAMES = labels
    self.FULL_LABEL_MAP = np.arange(len(self.LABEL_NAMES)).reshape(len(self.LABEL_NAMES), 1)
    self.FULL_COLOR_MAP = label_to_color_image(self.FULL_LABEL_MAP)


  def run(self, image):
    """Runs inference on a single image.

    Args:
      imgPath: Path for image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, PIL_image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    ###KEEP ONLY PERSON AND BACKGROUND -> 15 & 0 
    cats = np.unique(seg_map)
    np.place(seg_map,np.logical_and(seg_map != 15,seg_map != 0), 0)
    ###END###
    return resized_image, seg_map, cats

  def vis_segmentation(self, image, seg_map):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
        self.FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), self.LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap

def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


  # if __name__ == "__main__":
  #     LABEL_NAMES = np.asarray([
  #         'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
  #         'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
  #         'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
  #     ])

  #     FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
  #     FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
  #     model_path = '/Users/Alexandros/Desktop/issel/scripts/colorSpecification/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz'
  #     odapi = DeepLabModel(tarball_path=model_path)
  #     image = Image.open('/Users/Alexandros/Desktop/issel/scripts/miniDataset/bigone.jpeg')
  #     image, seg_map, cats = odapi.run(image)
  #     vis_segmentation(image, seg_map)
  #     pass



'''
    threshold = 0.7
    cap = cv2.imread('/Users/Alexandros/Desktop/issel/scripts/miniDataset/rac3.jpeg')
    img = cv2.resize(cap, (1280, 720))
    boxes, scores, classes, num = odapi.processFrame(img)
    for i in range(3):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
    plt.imshow(img)
    plt.show()
    pass
'''


