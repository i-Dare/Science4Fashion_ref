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


os.environ['KMP_DUPLICATE_LIB_OK']='True'

#https://stackoverflow.com/questions/9694165/convert-rgb-color-to-english-color-name-like-green-with-python
def get_colour_nameDetailed(rgb_triplet):
    min_colours = {}
    for key, name in mc.CSS4_COLORS.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(name)
        rd = (r_c - rgb_triplet[0]) ** 2
        gd = (g_c - rgb_triplet[1]) ** 2
        bd = (b_c - rgb_triplet[2]) ** 2
        min_colours[(rd + gd + bd)] = key
    return min_colours[min(min_colours.keys())]

def get_colour_name(rgb_triplet):
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
    return min_colours[min(min_colours.keys())]

def colorExtraction(image, colorRGBDF, colorDF, imgPath):
    # Initialize Cloth seperation module
    cloth = Cloth(imgPath, imgBGR=image)

    # Check for skin in the image
    kernel = np.ones((5,5), np.uint8)
    skin_erode_mask = cv2.erode(cloth.get_ycrcb_mask(), kernel, iterations=5)
    if skin_erode_mask.sum() > 400:
        _, cloth.segBackgroundMask, cloth.catsSegment  = odapi.run(imgPath)
        cloth.discardExtraCats()
        cloth.skinExtraction()    

    cloth.combineMasks()
    try:
        _, clothImg2D = imageUtils.reshapeDim(cloth.clothMask, cloth.clothImg)
        cloth.extractColor(clothImg2D)
    except:
        logger.info('Failed to extract color informantion for image %s' % imgPath)
        # In case of an error color RGB = (-1, -1, -1)
        color_fail = -1 * np.ones(3, dtype=int)
        cloth.colors = [(0., color_fail)] * 5

    # Save color information to database by updating ProductColor, ColorRGB and Product tables
    # DataFrame for ProductColor table ('Product','ColorRGB','Percentage','Ranking')
    colorCols = ['Product','ColorRGB','Percentage','Ranking']
    newEntryColorDF = pd.DataFrame(columns=colorCols)

    for ranking in range(5):
        color = cloth.colors[ranking][1].tolist()
        colorPercentage = cloth.colors[ranking][0]
        # Search if the color already exists in the ColorRGB table
        colorList = colorRGBDF[['Red','Green','Blue']].values.tolist()
        
        if color not in colorList: # Check if empty list so there is no match
            colorName = get_colour_name(color)
            colorNameDetails = get_colour_nameDetailed(color) 
            colorRow = color + [colorName] + [colorNameDetails]
            colorRGBCols = ['Red','Green','Blue','Label','LabelDetailed']
            colorSeries = pd.Series({column:value for column,value in zip(colorRGBCols, colorRow)})
            colorRGBDF = colorRGBDF.append(colorSeries, ignore_index=True)
            # DataFrame for ColorRGB table ('Red','Green','Blue','Label','LabelDetailed')                    
            newEntryColorRGBDF = pd.DataFrame(columns=colorRGBCols)
            newEntryColorRGBDF = newEntryColorRGBDF.append(colorSeries, ignore_index=True)
            newEntryColorRGBDF.to_sql('ColorRGB', schema='dbo', con = engine, if_exists = 'append', index = False)
            # newEntryColorRGBDF.to_sql('ColorRGB', con = engine, if_exists = 'append', index = False)
            logger.info('Adding color \"%s\" - \"%s\" %s in ColorRGB table' % (colorName, colorNameDetails, str(color)))
            colorRGBDF = pd.read_sql_query(colorRGBQuery, engine)
            colID = colorRGBDF['Oid'].values[-1]
        else: # not empty so there is a match
            colID = colorRGBDF.loc[colorList.index(color), 'Oid']

        newEntryColorDF.loc[ranking] = [row['Oid'][0]] + [colID] + [colorPercentage] + [ranking + 1]

    newEntryColorDF.to_sql('ProductColor', schema='%s.dbo' % dbName, con = engine, if_exists = 'append', index = False)
    # newEntryColorDF.to_sql('ProductColor', con = engine, if_exists = 'append', index = False)    
    return colorRGBDF, colorDF

if __name__ == '__main__':
    # Begin Counting Time
    start_time = time.time() 
    user = sys.argv[1]
    logging = S4F_Logger('ColorAnnotationLogger', user=user)
    logger = logging.logger
    helper = Helper(logging)

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
    query = '''SELECT *
                FROM %s.dbo.Product AS PR
                LEFT JOIN %s.dbo.ProductColor AS PC
                ON PR.Oid=PC.Product
                WHERE PC.Oid IS NULL''' % (str(dbName), str(dbName))
    # query = '''SELECT *
    #            FROM public."Product" AS PR
    #            LEFT JOIN public."ProductColor" AS PC
    #            ON PR."Oid"=PC."Product"
    #            WHERE PC."Product" IS NULL'''
    productDF = pd.read_sql_query(query, engine)

    #Colors dataframe
    for _, row in productDF.iterrows():
        #Read Color and ColorRGB from database
        colorQuery = '''SELECT * FROM %s.dbo.ProductColor''' % dbName
        # colorQuery = '''SELECT * FROM public."ProductColor"''' 
        colorDF = pd.read_sql_query(colorQuery, engine)
        colorRGBQuery = '''SELECT * FROM %s.dbo.ColorRGB''' % dbName
        # colorRGBQuery = '''SELECT * FROM public."ColorRGB"'''
        colorRGBDF = pd.read_sql_query(colorRGBQuery, engine)
        # Image path
        imgPath = row['Photo']
        try:
            if os.path.exists(imgPath):
                logger.info('Processing image: %s' % imgPath)
                # Open image for unicode file paths
                imgStream = open(imgPath, "rb")
                imgArray = np.asarray(bytearray(imgStream.read()), dtype=np.uint8)
                image = cv2.imdecode(imgArray, cv2.IMREAD_UNCHANGED)
                colorRGBDF, colorDF = colorExtraction(image, colorRGBDF, colorDF, imgPath)
            else:
                logger.info('Cannot find image with ID %s at path %s' % (row['Oid'], imgPath))
                imageBlob = row['Image']
                image = helper.convertBlobToImage(imageBlob)
                colorRGBDF, colorDF = colorExtraction(image, colorRGBDF, colorDF, 'Extracted image %s' % row['Oid'])
        except:
            logger.warning('Warning: No color information for image %s' % row['URL'])
    # End Counting Time
    logger.info("--- %s seconds ---" % (time.time() - start_time))
