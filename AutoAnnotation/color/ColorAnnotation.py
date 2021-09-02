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
from collections.abc import Iterable
import logging

from core.helper_functions import *
import core.config as config
from core.logger import S4F_Logger
from AutoAnnotation.color import cloth, imageUtils, segmentation
from AutoAnnotation.color.cloth import Cloth
from core.query_manager import QueryManager

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class ColorAnnotator():
    '''
    Rensponsible for color annotation
    '''
    def __init__(self, user, *oids, loglevel=config.DEFAULT_LOGGING_LEVEL):
        self.user = user
        self.oids = oids
        self.loglevel = loglevel
        self.logging = S4F_Logger('ColorAnnotationLogger', user=user, level=self.loglevel)
        self.logger = self.logging.logger
        self.helper = Helper(self.logging)
        self.db_manager = QueryManager(user=self.user)   

        modelPath = config.COLOR_MODELPATH
        #Segmentation Background & Person
        labels = np.asarray(config.CLASSES)
        self.odapi = segmentation.DeepLabModel(tarball_path=modelPath, labels=labels)
        
        # Prepare query
        dbName = config.DB_NAME
        if len(oids) > 0 and isinstance(oids, Iterable):
            where = ' OR '.join(['PR.Oid=%s' % i for i in  oids])
            query = '''SELECT PR.Oid, PR.Image, PR.ImageSource, PR.Photo 
                        FROM %s.dbo.Product AS PR
                        LEFT JOIN %s.dbo.ProductColor AS PC
                        ON PR.Oid=PC.Product
                        WHERE %s''' % (str(dbName), str(dbName), where)

        else:        
            query = '''SELECT PR.Oid, PR.Image, PR.ImageSource, PR.Photo 
                        FROM %s.dbo.Product AS PR
                        LEFT JOIN %s.dbo.ProductColor AS PC
                        ON PR.Oid=PC.Product
                        WHERE PC.Oid IS NULL''' % (str(dbName), str(dbName))
        # Read data from database
        self.products_df = self.db_manager.runSimpleQuery(query, get_identity=True)
        self.products_df = self.products_df.drop_duplicates().reset_index()
    
    def execute_annotation(self,):
        start_time = time.time() 
        self.logger.info("Executing color annotation for %s unlabeled products" % len(self.products_df))
        #Colors dataframe
        for _, row in self.products_df.iterrows():
            productID = row['Oid']
            # Image source
            imgSrc = row['Photo']
            # try:
            if os.path.exists(imgSrc):            
                # Open image for unicode file paths
                image = self.helper.openUnicodeImgPath(imgSrc)
            else:
                imageBlob = row['Image']
                image = self.helper.convertBlobToImage(imageBlob)
                # If image fails to load from binary, retrieve it from the image URL
                if image is None:
                    image = self.helper.getWebImage(row['ImageSource'])
                imgSrc = 'Extracted image %s' % productID
            _ = self.colorExtraction(image, imgSrc, row)

        # End Counting Time
        self.logger.info("--- Finished color annotation of %s records in %s seconds ---" % (len(self.products_df), 
                round(time.time() - start_time, 2)))
        self.logger.close()

    def colorExtraction(self, image, imgSrc, row):
        productID = row['Oid']
        if image is None:
            self.logger.warning('Failed to load image for Product with Oid %s' % productID, extra={'Product': productID})
            return -1
        self.logger.debug('Processing image of Product with Oid %s' % productID, extra={'Product': productID})
        # Initialize Cloth seperation module
        cloth = Cloth(imgSrc, imgBGR=image)

        # Check for skin in the image
        kernel = np.ones((5,5), np.uint8)
        skin_erode_mask = cv2.erode(cloth.get_ycrcb_mask(), kernel, iterations=5)
        if skin_erode_mask.sum() > 400:
            pil_image = self.helper.convertCVtoPIL(image)
            _, cloth.segBackgroundMask, cloth.catsSegment  = self.odapi.run(pil_image)
            cloth.discardExtraCats()
            cloth.skinExtraction()    

        cloth.combineMasks()
        try:
            _, clothImg2D = imageUtils.reshapeDim(cloth.clothMask, cloth.clothImg)
            cloth.extractColor(clothImg2D)
        except Exception as e:
            self.logger.warn_and_trace(e, extra={'Product': productID})
            self.logger.warning('Failed to extract color informantion for image %s' % imgSrc, extra={'Product': productID})
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
            productID = row['Oid']
            self.logger.debug('Captured color \"%s\" - \"%s\" %s for product %s' % (colorName, 
                    colorNameDetailed, str(color), productID), extra={'Product': productID})
            newEntryColorRGB_df = self.db_manager.runCriteriaInsertQuery(uniq_params=uniq_params, 
                                                                params=params, 
                                                                get_identity=True)

            # Prepare insert query for captured color in 'ProductColor' table
            params = {'Ranking': ranking+1, 'Percentage': colorPercentage, 
                    'ColorRGB': newEntryColorRGB_df.loc[0, 'Oid'], 'Product': productID,
                    'table': 'ProductColor'}
            self.db_manager.runInsertQuery(params)
            


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



if __name__ == '__main__':
    user = sys.argv[1]
    oids = sys.argv[2:-1]
    loglevel = sys.argv[-1]
    color_annotator = ColorAnnotator(user, *oids, loglevel=loglevel)
    color_annotator.execute_annotation()


    
