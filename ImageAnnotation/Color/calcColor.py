import sys
from PIL import Image
import cv2
import os
import numpy as np
import pandas as pd
import sqlalchemy
import webcolors
import matplotlib.colors as mc
from matplotlib import pyplot as plt

import helper_functions
import config

from cloth import Cloth
import imageUtils
import segmentation
os.environ['KMP_DUPLICATE_LIB_OK']='True'
CSS4LIST = mc.CSS4_COLORS

#https://stackoverflow.com/questions/9694165/convert-rgb-color-to-english-color-name-like-green-with-python
def get_colour_nameDetailed(rgb_triplet):
    min_colours = {}
    for key, name in CSS4LIST.items():
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
    return min_colours[min(min_colours.keys())]


if __name__ == '__main__':
    #Connect to database with sqlalchemy
    engine = helper_functions.ENGINE
    dbName = helper_functions.DB_NAME
    
    modelPath = config.MODELPATH
    #Segmentation Background & Person
    labels = np.asarray(config.CLASSES)
    odapi = segmentation.DeepLabModel(tarball_path=modelPath, labels=labels)

    #Read data from database
#     query = '''SELECT *
#                 FROM %s.dbo.Product AS PR
#                 LEFT JOIN %s.dbo.ProductColor AS PC
#                 ON PR.Oid=PC.Product
#                 WHERE PC.Oid IS NULL''' % (str(dbName), str(dbName))
    query = '''SELECT *
               FROM public."Product" AS PR
               LEFT JOIN public."ProductColor" AS PC
               ON PR."Oid"=PC."Product"
               WHERE PC."Product" IS NULL'''
    productDF = pd.read_sql_query(query, engine)

    #Create CSV to save the colors
    #Colors dataframe
    for rows in range(len(productDF))[:100]:
        #Read Color and ColorRGB from database
#         colorQuery = '''SELECT * FROM %s.dbo.ProductColor''' % dbName
        colorQuery = '''SELECT * FROM public."ProductColor"''' 
        colorDF = pd.read_sql_query(colorQuery, engine)
#         colorRGBQuery = '''SELECT * FROM %s.dbo.ColorRGB''' % dbName
        colorRGBQuery = '''SELECT * FROM public."ColorRGB"'''
        colorRGBDF = pd.read_sql_query(colorRGBQuery, engine)
        # Image path
        imgPath = str(productDF.loc[rows,'Photo'])
        if os.path.exists(imgPath):
            print('Processing image: %s' % imgPath)
            # Open image for unicode file paths
            imgStream = open(imgPath, "rb")
            imgArray = np.asarray(bytearray(imgStream.read()), dtype=np.uint8)
            image = cv2.imdecode(imgArray, cv2.IMREAD_UNCHANGED)
                        
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
                print('Failed to extract color informantion for image %s' % imgPath)
                # In case of an error color RGB = (-1, -1, -1)
                color_fail = -1 * np.ones(3, dtype=int)
                cloth.colors = [(0., color_fail)] * 5

            # Save color information to database by updating ProductColor, ColorRGB and Product tables
            # DataFrame for Color table ("ProductNo","ColorID","Percentage","Ranking")
            colorCols = ['Product','ColorRGP','Percentage','Ranking']
            newEntryColorDF = pd.DataFrame(columns=colorCols)
            colorRGBCols = ['Red','Green','Blue','Label','LabelDetailed']
            newEntryColorRGBDF = pd.DataFrame(columns=colorRGBCols)

            for ranking in range(5):
                color = cloth.colors[ranking][1].tolist()
                colorPercentage = cloth.colors[ranking][0]
                # Search if the color already exists in the ColorRGB table
                position = [index+1 if list(row) == color else None for index, row in colorRGBDF[['Red','Green','Blue']].iterrows()]
                res = [pos for pos in position if pos]
                if not res: # Check if empty list so there is no match
                    colorName = get_colour_name(color)
                    colorNameDetails = get_colour_nameDetailed(color) 
                    colorRow = color + [colorName] + [colorNameDetails]
                    colorSeries = pd.Series({column:value for column,value in zip(newEntryColorRGBDF.columns, colorRow)})
                    
                    colID = colorRGBDF.shape[0] + 1
                    colorRGBDF = colorRGBDF.append(colorSeries, ignore_index=True)
                    newEntryColorRGBDF = newEntryColorRGBDF.append(colorSeries, ignore_index=True)
                    print('Adding color \"%s\" - \"%s\" %s in ColorRGB table' % (colorName, colorNameDetails, str(color)))
                else: # not empty so there is a match
                    colID = colorRGBDF.loc[res.pop() - 1, 'Oid']

                newEntryColorDF.loc[ranking] = [productDF.loc[rows,'Oid'][0]] + [colID] + [colorPercentage] + [ranking + 1]
            
#             newEntryColorRGBDF.to_sql('ColorRGB', schema='dbo', con = engine, if_exists = 'append', index = False)
            newEntryColorRGBDF.to_sql('ColorRGB', con = engine, if_exists = 'append', index = False)
#             newEntryColorDF.to_sql('ProductColor', schema='dbo', con = engine, if_exists = 'append', index = False)
            newEntryColorDF.to_sql('ProductColor', con = engine, if_exists = 'append', index = False)

