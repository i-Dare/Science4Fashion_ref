import os
import json
import sqlalchemy

#
########### Universal variables ###########
#
CWD = os.getcwd()
PROJECT_HOME = os.environ['PROJECT_HOME']
PROJECT_CONFIG = os.path.join(PROJECT_HOME, 'config.json')
#PROJECT_CONFIG = os.path.join(PROJECT_HOME, 'config_test.json')
# Get project's configuration file
with open(PROJECT_CONFIG) as f:
    config = json.load(f)
DB_CONNECTION = config['db_connection']
DB_NAME = config['db_name']
ENGINE = sqlalchemy.create_engine(DB_CONNECTION + DB_NAME)

UPDATESQLQUERY = """
    UPDATE Product
    SET Product.ProductCategory = temp_table.ProductCategory, Product.ProductSubcategory = temp_table.ProductSubcategory, Product.Length = temp_table.Length, Product.Sleeve = temp_table.Sleeve, Product.CollarDesign = temp_table.CollarDesign, Product.NeckDesign = temp_table.NeckDesign, Product.Fit = temp_table.Fit
    FROM temp_table
    WHERE Product.Oid = temp_table.Oid;
    DROP TABLE temp_table;
"""
# UPDATESQLQUERY = """
#     UPDATE "Product" 
#     SET "ProductCategory" = temp_table."ProductCategory", "ProductSubcategory" = temp_table."ProductSubcategory", "Length" = temp_table."Length", "Sleeve" = temp_table."Sleeve", "CollarDesign" = temp_table."CollarDesign", "NeckDesign" = temp_table."NeckDesign", "Fit" = temp_table."Fit"
#     FROM temp_table 
#     WHERE public."Product"."Oid" = public."temp_table"."Oid";
#     DROP TABLE public."temp_table";
# """


#
########### Modules paths ###########
#
# Setup paths for the various modules
TEXT_MINING = os.path.join(PROJECT_HOME, 'TextMining')
CLUSTERING = os.path.join(PROJECT_HOME, 'Clustering')
IMAGE_ANNOTATION = os.path.join(PROJECT_HOME, 'ImageAnnotation')
RECOMMENDER = os.path.join(PROJECT_HOME, 'Recommender')
WEB_CRAWLERS = os.path.join(PROJECT_HOME, 'WebCrawlers')
RESOURCESDIR = os.path.join(PROJECT_HOME, config['resources']['resourcesDir'])
IMAGEDIR = os.path.join(PROJECT_HOME, config['resources']['resourcesDir'], 'images')
MODELSDIR = os.path.join(PROJECT_HOME, RESOURCESDIR,config['resources']['models']['modelsDir'])
COLOR_MODEL_DIR = os.path.join(PROJECT_HOME, RESOURCESDIR, MODELSDIR, config['resources']['models']['color_model']['directory'])
CLUSTERING_MODEL_DIR = os.path.join(PROJECT_HOME, RESOURCESDIR, MODELSDIR, config['resources']['models']['clustering_model']['directory'])
PRODUCT_ATTRIBUTE_MODEL_DIR = os.path.join(PROJECT_HOME, RESOURCESDIR, MODELSDIR, config['resources']['models']['product_attribute_model']['directory'])

#
########### WebCrawler variables ###########
#
# DB Initiation
MAIN_DB_TABLES = config['main_db_tables']

#
########### WebCrawler variables ###########
#
# Pinterest fields
PINTEREST_USERNAME = config['social_media']['pinterest_username']
PINTEREST_PASSWORD = config['social_media']['pinterest_password']
# Instagram fields
INSTAGRAM_USERNAME = config['social_media']['pinterest_username']
INSTAGRAM_PASSWORD = config['social_media']['pinterest_password']

#
########### TextMining variables ###########
#
PRODUCT_ATTRIBUTES_PATH = os.path.join(RESOURCESDIR, config['resources']['product_attributes'])
SHEETNAME = 'product_attributes_sheet'
FASHION_WORDS = os.path.join(RESOURCESDIR, config['resources']['fashion_words'])
PRODUCT_ATTRIBUTES = ['ProductCategory', 'ProductSubcategory', 'Length', 'Sleeve', 'CollarDesign', 'NeckDesign', 'Fit']

#
########### ImageAnnotation variables ###########
#
# calcColor variables
COLOR_MODELPATH = os.path.join(COLOR_MODEL_DIR, config['resources']['models']['color_model']['model'])
CLASSES = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv' 
         ]
#        
# Attribute prediction variables
ATTRIBUTE_COLUMNS = ['Length', 'Sleeve', 'CollarDesign', 'NeckDesign', 'Fit']
#Device to test
DEVICE = 'cpu'
# Dictionaries to map the predicted label to the possible labels and the respective models
# -1 is the Strapless we don't know how to handle it yet
#Neckline
DICTNECKLINE = {0:2,1:7, 2:4, 3:6, 4:1, 5:3, 6:5}
MODELNECKLINE = config['resources']['models']['product_attribute_model']['model']['neckline']
#Sleeve
DICTSLEEVE = {0:8, 1:2, 2:6, 3:1, 4:4}
MODELSLEEVE = config['resources']['models']['product_attribute_model']['model']['sleeve']
#Length
DICTLENGTH = {0:4, 1:2, 2:3, 3:1}
MODELLENGTH = config['resources']['models']['product_attribute_model']['model']['length']
#Collar
DICTCOLLAR = {0:4, 1:-1, 2:1}
MODELCOLLAR = config['resources']['models']['product_attribute_model']['model']['collar']
#Fit
DICTFIT = {0:2, 1:1, 2:3, 3:4}
MODELFIT = config['resources']['models']['product_attribute_model']['model']['fit']

#
########### Clustering variables ###########
#
# KModes clustering variables
# Number of clusters
CLUSTERING_PRODUCT_ATTRIBUTES = ['RetailPrice', 'Gender'] + PRODUCT_ATTRIBUTES
MODELCLUSTERING = config['resources']['models']['clustering_model']['model']
INITKMODES = 'Cao'
UPDATE_CLUSTERS_QUERY = """
    UPDATE Cluster
    SET Cluster.Cluster = temp_table.Cluster, Cluster.ProductCategory = temp_table.ProductCategory, Cluster.ProductSubcategory = temp_table.ProductSubcategory, Cluster.Gender = temp_table.Gender,
        Cluster.LifeStage = temp_table.LifeStage, Cluster.Length = temp_table.Length, Cluster.Sleeve = temp_table.Sleeve, Cluster.CollarDesign = temp_table.CollarDesign, Cluster.NeckDesign = temp_table.NeckDesign, Cluster.Fit = temp_table.Fit
    FROM Cluster
    WHERE Cluster.Cluster = temp_table.Cluster
"""
# K-Modes clustering
# UPDATE_CLUSTERS_QUERY = """
#     UPDATE "Cluster" 
#     SET "Oid" = "temp_table"."Cluster", "ProductCategory" = "temp_table"."ProductCategory", "ProductSubcategory" = "temp_table"."ProductSubcategory", "Gender" = "temp_table"."Gender",
#         "LifeStage" = "temp_table"."LifeStage", "Length" = "temp_table"."Length", "Sleeve" = "temp_table"."Sleeve", "CollarDesign" = "temp_table"."CollarDesign", "NeckDesign" = "temp_table"."NeckDesign", "Fit" = "temp_table"."Fit"
#     FROM temp_table 
#     WHERE public."Cluster"."Oid" = "temp_table"."Cluster"
# """

UPDATE_PRODUCT_CLUSTERS_QUERY = """
    UPDATE Product
    SET Product.Cluster = temp_table.Cluster
    FROM temp_table
    WHERE Product.Oid = temp_table.Oid;
    DROP TABLE temp_table;
"""
# UPDATE_PRODUCT_CLUSTERS_QUERY = """
#     UPDATE "Product"
#     SET "Cluster" = "temp_table"."Cluster"
#     FROM "temp_table"
#     WHERE "Product"."Oid" = "temp_table"."Oid";
#     DROP TABLE public."temp_table";
# """

# Consensus Clustering variables
FAMD_COMPONENTS = config['clustering']['famd_components']
LINKAGE = config['clustering']['linkage']
DISTANCE_THRESHOLD = config['clustering']['distance_threshold']
UPDATE_PRODUCT_CONSENSUS_CLUSTERS_QUERY = """
    UPDATE Product
    SET Product.ConsensusCluster = temp_table.ConsensusCluster
    FROM temp_table
    WHERE Product.Oid = temp_table.Oid;
    DROP TABLE temp_table;
"""
# UPDATE_PRODUCT_CONSENSUS_CLUSTERS_QUERY = """
#     UPDATE "Product"
#     SET "ConsensusCluster" = "temp_table"."ConsensusCluster"
#     FROM "temp_table"
#     WHERE "Product"."Oid" = "temp_table"."Oid";
#     DROP TABLE public."temp_table";
# """