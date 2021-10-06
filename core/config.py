import os
import json
import sqlalchemy

#
########### Universal variables ###########
#
CWD = os.getcwd()
PROJECT_HOME = os.environ['PROJECT_HOME']
PROJECT_CONFIG = os.path.join(PROJECT_HOME, 'config.json')

# Get project's configuration file
with open(PROJECT_CONFIG) as f:
    config = json.load(f)
DB_CONNECTION = config['db_connection']
DB_NAME = config['db_name']
ENGINE = sqlalchemy.create_engine(DB_CONNECTION + DB_NAME)
DEFAULT_LOGGING_LEVEL = config['default_logging_level']
BATCH_STEP = config['batch_step']
FINANCIAL_ATTRIBUTES = config['financial_attributes']

#
########### Modules paths ###########
#
# Setup paths for the various modules
AUTOANNOTATION = os.path.join(PROJECT_HOME, 'AutoAnnotation')
CLUSTERING = os.path.join(PROJECT_HOME, 'Clustering')
RECOMMENDER = os.path.join(PROJECT_HOME, 'Recommender')
WEB_CRAWLERS = os.path.join(PROJECT_HOME, 'WebCrawlers')
RESOURCESDIR = os.path.join(PROJECT_HOME, config['resources']['resourcesDir'])
DATADIR = os.path.join(PROJECT_HOME, RESOURCESDIR, config['resources']['data']['directory'])
IMAGEDIR = os.path.join(PROJECT_HOME, config['resources']['resourcesDir'], 'images')
MODELSDIR = os.path.join(PROJECT_HOME, RESOURCESDIR,config['resources']['models']['modelsDir'])
COLOR_MODEL_DIR = os.path.join(PROJECT_HOME, RESOURCESDIR, MODELSDIR, config['resources']['models']['color_model']['directory'])
CLUSTERING_MODEL_DIR = os.path.join(PROJECT_HOME, RESOURCESDIR, MODELSDIR, config['resources']['models']['clustering_models']['directory'])
PRODUCT_ATTRIBUTE_MODEL_DIR = os.path.join(PROJECT_HOME, RESOURCESDIR, MODELSDIR, config['resources']['models']['product_attribute_model']['directory'])
TEXT_DESCRIPTOR_MODEL_DIR = os.path.join(PROJECT_HOME, RESOURCESDIR, MODELSDIR, config['resources']['models']['text_descriptor_model']['directory'])
INCREMENTAL_LEARNING_MODEL_DIR = os.path.join(PROJECT_HOME, RESOURCESDIR, MODELSDIR, config['resources']['models']['rating_model']['directory'])

#
########### Logging variables ###########
#
DEFAULT_USER =  config['default_user']

#
########### DB variables ###########
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
# Connection timeout
CRAWLER_TIMEOUT = config['crawler_timeout']
# Asos cookie
ASOS_COOKIE={'geocountry':'GR', 'bm_sz':'96EB6098A81062F612CC102536790284~YAAQpDMTAsgdcbJ6AQAA17XUYwxRz16lSqDTIeQCKCHpbAPx/hwmvC1SP9746T8IHwew3zFL5I4rt/j+YHGKD6TwfbrSOGQ8eIdDC9mcn16MvNtfCsoJdZODuicGyCYYwGsDnBgLvbFOJesFibFDdjjWZ+kwqqAxSNVO8AWh/H45ZqivR0ywAm51ucqsKNy9yDWEdbcClszRf+fNPV0lN6mBmMGp1adVLd9u2/MwGBJqbskOhAd5Qya/cPUWMGbJ+1eys9/AViYn7xKbBEXi6hvCt4RXOTRNu9NOw6vlBY0S~4276790~4339512', 'keyStoreDataversion':'hnm9sjt-28', 'browseCountry':'GR', 'browseCurrency':'EUR', 'browseLanguage':'en-GB', 'browseSizeSchema':'EU', 'storeCode':'ROE', 'currency':'19', 'featuresId':'ec2ee2ca-2ecb-4e35-9f51-77b8d206c68c', 'ak_bmsc':'DB5D793307E1879A9922C1D18D044DD5~000000000000000000000000000000~YAAQpDMTAv0dcbJ6AQAA17/UYwxYHkrf5gT08j2QC1z0abHshw/Kz49gYWj68zkkIub20R9VyUcXssmGNoWOQhZ4Dn2gYLMhCbZbnGbgmcXyYQq/SEsbmsr363LTupdEMbZ9Xu0EeRRC1nI+UEbjJu3E7fvguRtZY/xuZJVmunq2sYJ86FfrbNkp/Ey6a2mlFoQX8OA0JNTZuupcnUAPWoCEnK5Ch0NhFcatMbY1yfAsITxXjOYMFohE2852QZ4yNlz5ZUHHvuofWrJ9WUewgW643CG0DY1q8ZdqhRwBwmLnqYzUYzSLe8xZi24c68NKpsJ8+8DndgrVrR1dhZ1mz6JQ7hbRu5AUdPEVXCM6VOuoGGB46tU15qU/dZ8LbVVT5g52VjqSSQ==', 'asos-b-sdv629':'hnm9sjt-28',  'plp_columsCount':'fourColumns',  'AMCVS_C0137F6A52DEAFCC0A490D4C%40AdobeOrg':'1',  'AMCV_C0137F6A52DEAFCC0A490D4C%40AdobeOrg':'-1303530583%7CMCMID%7C20218587709322395641781164793527994063%7CvVersion%7C3.3.0%7CMCAID%7CNONE%7CMCOPTOUT-1629474693s%7CNONE',  '_s_fpv':'true',  's_cc':'true',  's_fid':'044921F6E6A75B5E-0301A8C8590F2FEC',  's_pers':'%20s_vnum%3D1630443600595%2526vn%253D1%7C1630443600595%3B%20s_invisit%3Dtrue%7C1629470666175%3B%20s_nr%3D1629468866178-New%7C1661004866178%3B%20gpv_e47%3Dno%2520value%7C1629470666180%3B%20gpv_p10%3Ddesktop%2520roe%257Csearch%2520page%257Csuccessful%2520search%2520refined%2520page%25201%7C1629470666182%3B',  'asos-perx':'b93eb4197c0e4bdb940914d5ffe69df6||5472ef464c734a4dade1db6712f33905',  '_abck':'94BD150A7B8423688670D8244A365F09~-1~YAAQVEUVAtVLs7N6AQAAmuzqYwb/26tsqi8xiv2JSoruCyGp5CHEXr7uTFaojNNelagnjeS6Zn6o0DmdEKrI/lyul8H8OR/vfRPnsQD2CWnBU7xa54puf4gZq6goF4t4DdM3UYQL1yVBRvZgpk+Q+oneVaj2sNl4+LTHtEieyuMrJ5o0HFyk2LK4SLzd3bjznD4VRDXAMfnsqBIKMOGVHSrhHhzNCVqlu1mpM8wBpIJTPQoxdGaPuS8R215P+WZPW/IkgahRKm/ngu8RkLwM/bnSEprE9n3bkqN7RZRsblcKX69odMAgo3fSJ1O4Kp1WkAdiIPK6ciTioKg/1qJWJYvATDv/DvSKGJWvGwLFGncelI8l4kJn6Np25CG6tSyx7ZhjKstU5kY=~-1~-1~-1'}

#
########### TextMining variables ###########
#
PRODUCT_ATTRIBUTES_PATH = os.path.join(RESOURCESDIR, config['resources']['product_attributes'])
SHEETNAME = 'product_attributes_sheet'
FASHION_WORDS = os.path.join(RESOURCESDIR, config['resources']['fashion_words'])
PRODUCT_ATTRIBUTES = ['ProductCategory', 'ProductSubcategory', 'Length', 'Sleeve', 'CollarDesign', 'NeckDesign', 'Fit']

#
########### AutoAnnotation variables ###########
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
MODEL_NECKLINE = config['resources']['models']['product_attribute_model']['model']['neckline']
#Sleeve
DICTSLEEVE = {0:8, 1:2, 2:6, 3:1, 4:4}
MODEL_SLEEVE = config['resources']['models']['product_attribute_model']['model']['sleeve']
#Length
DICTLENGTH = {0:4, 1:2, 2:3, 3:1}
MODEL_LENGTH = config['resources']['models']['product_attribute_model']['model']['length']
#Collar
DICTCOLLAR = {0:4, 1:2, 2:1}
MODEL_COLLAR = config['resources']['models']['product_attribute_model']['model']['collar']
#Fit
DICTFIT = {0:2, 1:1, 2:3, 3:4}
MODEL_FIT = config['resources']['models']['product_attribute_model']['model']['fit']

#
########### Clustering variables ###########
#
# KModes clustering variables
# Number of clusters
CLUSTERING_PRODUCT_ATTRIBUTES = ['RetailPriceSoldRegular', 'Gender'] + PRODUCT_ATTRIBUTES
MODEL_CONSENSUS = config['resources']['models']['clustering_models']['consensus']
MODEL_KMEANS = config['resources']['models']['clustering_models']['kmeans']
MODEL_BIRCH = config['resources']['models']['clustering_models']['birch']
MODEL_FUZZYCMENS = config['resources']['models']['clustering_models']['fuzzycmeans']
MODEL_DBSCAN = config['resources']['models']['clustering_models']['dbscan']
MODEL_OPTICS = config['resources']['models']['clustering_models']['optics']
MODEL_BGM = config['resources']['models']['clustering_models']['bayesian_gaussian_mixture']
INITKMODES = 'Cao'

# Consensus Clustering variables
FAMD_COMPONENTS = config['clustering']['famd_repeats']
LINKAGE = config['clustering']['linkage']
DISTANCE_THRESHOLD = config['clustering']['distance_threshold']
SIMILARITY_MATRIX = os.path.join(DATADIR, config['resources']['data']['similarity_matrix'])

#
# Recommendation models
#
#
RATING_MODEL = config['resources']['models']['rating_model']['model']
IRRELEVANCE_MODEL = config['resources']['models']['irrelevance_model']['model']

########### Descriptor variables ###########
#
# Text descriptor model
MODEL_TEXT_DESCRIPTOR = config['resources']['models']['text_descriptor_model']['model']

# Saved data
TFIDF_VECTOR = os.path.join(DATADIR, config['resources']['data']['tfidf_data'])
