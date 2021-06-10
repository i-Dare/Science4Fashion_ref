import sys
import cv2
import os
import time
import sqlalchemy
import webcolors
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.colors as mc
from matplotlib import pyplot as plt
import logging

from core.helper_functions import *
import core.config as config
from core.logger import S4F_Logger
from ImageAnnotation.color import cloth, imageUtils, segmentation
from ImageAnnotation.color.cloth import Cloth
from core.query_manager import QueryManager

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#https://stackoverflow.com/questions/9694165/convert-rgb-color-to-english-color-name-like-green-with-python
def get_color_name_detailed(rgb_triplet):
    min_colours = {}
    for key, name in mc.CSS4_COLORS.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(name)
        rd = (r_c - rgb_triplet[0]) ** 2
        gd = (g_c - rgb_triplet[1]) ** 2
        bd = (b_c - rgb_triplet[2]) ** 2
        min_colours[(rd + gd + bd)] = key
    return min_colours[min(min_colours.keys())].lower()

def get_color_name(rgb_triplet):
    min_colours = {}
    for key, name in webcolors.CSS21_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - rgb_triplet[0]) ** 2
        gd = (g_c - rgb_triplet[1]) ** 2
        bd = (b_c - rgb_triplet[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    for key, name in mc.TABLEAU_COLORS.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(name)
        rd = (r_c - rgb_triplet[0]) ** 2
        gd = (g_c - rgb_triplet[1]) ** 2
        bd = (b_c - rgb_triplet[2]) ** 2
        min_colours[(rd + gd + bd)] = key.split(':')[1]
    return min_colours[min(min_colours.keys())].lower()

def colorExtraction(image):
    if image is None:
        logger.warning('Failed to load image for Product with Oid %s' % row['Oid'])
        return -1
    logger.info('Processing image of Product with Oid %s' % row['Oid'])
    # Initialize Cloth seperation module
    cloth = Cloth(imgSrc, imgBGR=image)

    # Check for skin in the image
    kernel = np.ones((5,5), np.uint8)
    skin_erode_mask = cv2.erode(cloth.get_ycrcb_mask(), kernel, iterations=5)
    if skin_erode_mask.sum() > 400:
        pil_image = helper.convertCVtoPIL(image)
        _, cloth.segBackgroundMask, cloth.catsSegment  = odapi.run(pil_image)
        cloth.discardExtraCats()
        cloth.skinExtraction()    

    cloth.combineMasks()
    try:
        _, clothImg2D = imageUtils.reshapeDim(cloth.clothMask, cloth.clothImg)
        cloth.extractColor(clothImg2D)
    except:
        logger.info('Failed to extract color informantion for image %s' % imgSrc)
        # In case of an error color RGB = (-1, -1, -1)
        color_fail = -1 * np.ones(3, dtype=int)
        cloth.colors = [(0., color_fail)] * 5

    # Save color information to ColorRGB and ProductColor tables
    for ranking in range(5):
        color = cloth.colors[ranking][1].tolist()
        colorPercentage = cloth.colors[ranking][0]
        # Get color name and detailed color name
        colorName = get_color_name(color)
        colorNameDetailed = get_color_name_detailed(color)

        # Prepare insert query for captured color in 'ColorRGB' table
        rgb_list = ['Red','Green','Blue']
        dict_keys = rgb_list + ['Label','LabelDetailed', 'Description']
        dict_values = color + [colorName] + [colorNameDetailed] + ['%s-%s' % (colorName, colorNameDetailed)]
        params = dict(zip(dict_keys, dict_values))
        uniq_params = dict(zip(rgb_list, color))
        params['table'] = uniq_params['table'] = 'ColorRGB'
        logger.info('Captured color \"%s\" - \"%s\" %s' % (colorName, 
                                                           colorNameDetailed, 
                                                           str(color)))
        newEntryColorRGB_df = db_manager.runCriteriaInsertQuery(uniq_params=uniq_params, 
                                                               params=params, 
                                                               get_identity=True)

        # Prepare insert query for captured color in 'ProductColor' table
        params = {'Ranking': ranking+1, 'Percentage': colorPercentage, 
                  'ColorRGB': newEntryColorRGB_df.loc[0, 'Oid'], 'Product': row['Oid'],
                  'table': 'ProductColor'}
        db_manager.runInsertQuery(params)

if __name__ == '__main__':
    # Begin Counting Time
    start_time = time.time() 
    user = sys.argv[1]
    logging = S4F_Logger('ColorAnnotationLogger', user=user)
    logger = logging.logger
    helper = Helper(logging)
    db_manager = QueryManager(user=user)    

    ### Read Table Products from S4F database ###
    logger.info('Loading Product table...')    
    #Connect to database with sqlalchemy
    engine = config.ENGINE
    dbName = config.DB_NAME
    
    modelPath = config.COLOR_MODELPATH
    #Segmentation Background & Person
    labels = np.asarray(config.CLASSES)
    odapi = segmentation.DeepLabModel(tarball_path=modelPath, labels=labels)

    #Read data from database
    query = '''SELECT PR.Oid, PR.Image, PR.ImageSource, PR.Photo 
                FROM %s.dbo.Product AS PR
                LEFT JOIN %s.dbo.ProductColor AS PC
                ON PR.Oid=PC.Product
                WHERE PC.Oid IS NULL''' % (str(dbName), str(dbName))
    product_df = db_manager.runSimpleQuery(query, get_identity=True)

    #Colors dataframe
    for _, row in product_df.iterrows():
        # Image source
        imgSrc = row['Photo']
        # try:
        if os.path.exists(imgSrc):            
            # Open image for unicode file paths
            image = helper.openUnicodeImgPath(imgSrc)
        else:
            imageBlob = row['Image']
            image = helper.convertBlobToImage(imageBlob)
            # If image fails to load from binary, retrieve it from the image URL
            if image is None:
                image = helper.getWebImage(row['ImageSource'])
            imgSrc = 'Extracted image %s' % row['Oid']
        _ = colorExtraction(image)

    # End Counting Time
    logger.info("--- %s seconds ---" % (time.time() - start_time))
