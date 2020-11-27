from pdf2image import convert_from_path
from PIL import Image
# For running inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


def run_detector(detector, img):
    #img = load_img(path)
    #img = tf.image.decode_jpeg(img, channels = 3)
    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)

    result = {key:value.numpy() for key,value in result.items()}
    return result

module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" 
detector = hub.load(module_handle).signatures['default']

pages = convert_from_path('/home/alexandros/Downloads/KIDS FASHION.pdf', 400, fmt='jpeg')
savedir = '/home/alexandros/Desktop/pdfimages/'

for indexPage, page in enumerate(pages):
    print(indexPage)
    count = 0
    im_width, im_height = page.size
    img = np.array(page) 
    res = run_detector(detector, img)
    print(res)
    #Filter Person
    keepIndices = []
    for index,value in enumerate(res['detection_class_labels']):
        if value == 69: # Person ID
            keepIndices.append(index)
    #print()
    newres = {key:value[keepIndices] for key, value in res.items()}
    '''
    #Filter Detection SCore
    keepIndices = []
    for index,value in enumerate(newres['detection_scores']):
        if value >= 0.01: # 
            keepIndices.append(index)
    newnewres = {key:value[keepIndices] for key, value in newres.items()}
    '''
    bboxes = newres["detection_boxes"]
    for bbox in bboxes:
        bbox = list(bbox)
        bbox = (bbox[1] * im_width, bbox[3] * im_width, 
                              bbox[0] * im_height, bbox[2] * im_height)
        temp = page.crop((bbox[0], bbox[2], bbox[1], bbox[3]))
        count += 1
        temp.save(savedir + str(indexPage) + str(count) + '.jpeg')

























