import os
import sqlalchemy
import json

#
########### Universal variables ###########
#
CWD = os.getcwd()
PROJECT_HOME = os.environ['PROJECT_HOME']
# PROJECT_CONFIG = os.path.join(PROJECT_HOME, 'config.json')
PROJECT_CONFIG = os.path.join(PROJECT_HOME, 'config_test.json')
# Get project's configuration file
with open(PROJECT_CONFIG) as f:
    config = json.load(f)
DB_CONNECTION = config['db_connection']
DB_NAME = config['db_name']
ENGINE = sqlalchemy.create_engine(DB_CONNECTION + DB_NAME)

# UPDATESQLQUERY = """
#     UPDATE Product
#     SET Product.ProductCategory = temp_table.ProductCategory, Product.ProductSubcategory = temp_table.ProductSubcategory, Product.Length = temp_table.Length, Product.Sleeve = temp_table.Sleeve, Product.CollarDesign = temp_table.CollarDesign, Product.NeckDesign = temp_table.NeckDesign, Product.Fit = temp_table.Fit
#     FROM temp_table
#     WHERE Product.Oid = temp_table.Oid;
#     DROP TABLE temp_table;
# """
UPDATESQLQUERY = """
    UPDATE "Product" 
    SET "ProductCategory" = temp_table."ProductCategory", "ProductSubcategory" = temp_table."ProductSubcategory", "Length" = temp_table."Length", "Sleeve" = temp_table."Sleeve", "CollarDesign" = temp_table."CollarDesign", "NeckDesign" = temp_table."NeckDesign", "Fit" = temp_table."Fit"
    FROM temp_table 
    WHERE public."Product"."Oid" = public."temp_table"."Oid";
    DROP TABLE public."temp_table";
"""
#
########### Modules paths ###########
#
# Setup paths for the various modules
TEXT_MINING = os.path.join(PROJECT_HOME, 'TextMining')
CLUSTERING = os.path.join(PROJECT_HOME, 'Clustering')
IMAGE_ANNOTATION = os.path.join(PROJECT_HOME, 'ImageAnnotation')
RECOMMENDER = os.path.join(PROJECT_HOME, 'Recommender')
WEB_CRAWLERS = os.path.join(PROJECT_HOME, 'WebCrawlers')

#
########### WebCrawler variables ###########
#
# Pinterest fields
PINTEREST_USERNAME = config['pinterest_username']
PINTEREST_PASSWORD = config['pinterest_password']

#
########### TextMining variables ###########
#
NRGATTRIBUTESPATH = TEXT_MINING +'\\data_exploration1.xlsx'
ORIGINALNRGATTRIBUTESPATH = TEXT_MINING + '\\data_exploration2.xlsx'
SHEETNAME = 'data_exploration2'
# DEEPFASHIONPATH = TEXT_MINING + '\\list_attr_cloth.txt'
NRGATTRIBUTES = ['ProductCategory', 'ProductSubcategory', 'Length', 'Sleeve', 'CollarDesign', 'NeckDesign', 'Fit']
# DEEPFASHIONATTRIBUTES = ['Texture','Fabric','Shape','Part','Style']
NRGATTRIBUTESID = ['ProductCategory', 'ProductSubcategory', 'Length', 'Sleeve', 'CollarDesign', 'NeckDesign', 'Fit']
# DEEPFASHIONATTRIBUTESID = ['Texture','Fabric','Shape','Part','Style']

#
########### ImageAnnotation variables ###########
#
# calcColor variables
COLOR_MODELPATH = os.path.join(IMAGE_ANNOTATION, 'Color', 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz')
CLASSES = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv' 
        ]
# Attribute prediction variables
ATTRIBUTE_COLUMNS = ['Length', 'Sleeve', 'CollarDesign', 'NeckDesign', 'Fit']
#Device to test
DEVICE = 'cpu'
#MODEL PATHS
ATTRIBUTE_MODELPATH = os.path.join(IMAGE_ANNOTATION, 'Prediction', 'models')

# Dictionaries to map the predicted label to the possible labels and the respective models
# -1 is the Strapless we don't know how to handle it yet
#Neckline
DICTNECKLINE = {0:2,1:7, 2:4, 3:6, 4:1, 5:3, 6:5}
MODELNECKLINE = "neckline.pkl"
#Sleeve
DICTSLEEVE = {0:8, 1:2, 2:6, 3:1, 4:4}
MODELSLEEVE = "sleeve.pkl"
#Length
DICTLENGTH = {0:4, 1:2, 2:3, 3:1}
MODELLENGTH = "length.pkl"
#Collar
DICTCOLLAR = {0:4, 1:-1, 2:1}
MODELCOLLAR = "collar.pkl"
#Fit
DICTFIT = {0:2, 1:1, 2:3, 3:4}
MODELFIT = "fit.pkl"
