# Science4Fashion Refactoring Repository

## Introduction
The Science4Fashion project aims at facilitating the design of clothing products, particularly in the field of product concept development, by providing personalized proposals to the designer as inspiration, by developing tool based on Artificial Intelligence methods. The tools will work in addition to existing support systems for the development of clothing products such as Life Cycle Management Systems, CAD (Graphics), 3D Modeling systems, commonly used by companies and designers.

The System is comprised of the following components:
1. [Data Collection](#1-data-collection)
2. [Data Annotation](#2-image-annotation)
3. [Clothing Recommender with User Feedback Component](#3-clothing-recommender-with-user-feedback)
4. [Science4Fashion Installation Guide](#4-science4Fashion-installation-guide)

## 1. Data Collection
With regards to data collection, the system targets a number of well known online resources and collects images, text and metadata according to a user defined query. There data sources can be grouped to website-based, that are accessed  via a number of web crawlers that collect the necessary information and social media-based that usually contain trending content that is rich but noisy. The data sources are referred  as `adapters`, and each adapter is responsible for a single data source. Namely, the web-based adapters are Asos, sOliver, and Zalando, whereas the social media-based are Instagram and Pinterest.

The data collection process is initiated by the user who provides a search term and a valid adaptor invoking the `crawl_search_wrapper.py` script. Apart from data retrieval, the script is also responsible to invoke the data annotation and clustering modules as well. Firstly, the contains all the implemented adapters and receives as arguments the name of the adapter, a search term and optionally the maximum number of results and the user ID. The search term should contain keywords that would otherwise be used as a query to a fashion retailers eshop or to a search engine and they should best describe the desired content. Afterwards, the selected adapter will query the respective source for the desired content and attempt to capture a number of attributes for each result. The product name, brand, image, description, genre, price and other metadata will be stored in the system's database. 

As soon as the adapter finishes retrieving the requested information from the source, the automated annotation process is initiated that will seek to infer specific clothing attributes regarding the color, neck-line, sleeves, length and fit of the clothing article. The annotation is driven by the retrieved image, the description and the metadata.

crawl_search_wrapper.py:
Wrapper script for website crawling and automatic annotation of the results. Receives as arguments the search terms, the adapter and the number of expected results. Is responsible for updating the S4F database with the search results and triggering the data annotation process. The annotation process consists of the following steps:
* Text based annotation
* Color annotation
* Apparel annotation
* ~~Clustering~~

#### Arguments:
```
* -i or -id (required): the CrawlSearch ID
* -l or --loglevel (optional): the logging level
```


#### Examples:
`$ python %PROJECT_HOME%/crawl_search_wrapper.py -i 13 -l debug`

Retrieves the CrawlSearch record with ID 13 and performs the data retrieval according to the record's fields (search term, number of results, adapter)

---

**NOTE:** Since Instagram uses two factor authentication, if the system attempts to access the service through the default credentials - user `idare.issel@gmail.com` - there is a high possibility that Instagram will refuse the connection. To bypass restriction, the user has to first log in through Firefox using these credentials, accept all cookies and then close the window without exiting Instagram. This way the crawler will fallback to logging in the service using an active session instead of creating a new one.

---


## 2. Image Annotation
The image annotation process is executed at the end of the adapter and is responsible for extracting the main product attributes. At this time, the system will attempt to capture the five most dominant colors of the product. Firstly, the product image is processed and the background is extracted. Secondly, the algorith discards any parts of the foreground that may contain skin color information, thus creating a mask for the clothing article. Finally, the color information inside the mask will be decomposed to the 5 most dominant colors and each color will be stored in the database along with the product ID.

Apart from the color information, the rest of the product's attributes regarding the collar style, length, neckline, sleeve type and general fit of the article are infered through a pretrained Deep Neural Network. The DNN is trained on on popular fashion datasets and employs adopted architectures such as VGG16, ResNet50 and ResNet50v2. 

The data annotation process is triggered from the user by providing appropriate arguments to invoke the data_annotation_wrapper.py script. Apart from data annotation, the script is also responsible for invoking the clustering module. 

data_annotation_wrapper.py:
Invokes the data annotation process by providing appropriate arguments. The annotation process consists of the following steps:
* Text based annotation
* Color annotation
* Apparel annotation
~~* Clustering~~

#### Arguments:
```
* -i or --id (optional): the product ID (“Oid”) list
* -u or -user (required): the user’s unique identifier
* -l or --loglevel (optional): the logging level
```

#### Examples:
```python %PROJECT_HOME%/data_annotation_wrapper.py -u 8D31B96A-02AC-4531-976F-A455686F8FE2 -i 4622 4623 4624 --l debug```

Annotates products with Oid "4624", "4623", and "4622" at debug level, expecting extensive event logging.

```python %PROJECT_HOME%/data_annotation_wrapper.py -u 8D31B96A-02AC-4531-976F-A455686F8FE2```

When no Oid argument (-i) is provided, the script annotates all non-annotated products at default logging level, expecting typical event logging.

## 3. Clothing Recommender with User Feedback
The fashion_recommendation.py script is responsible for triggering the execution of the Recommendation Module’s processes, as part of the Science4Fashion backend. The script handles recommendation production as a two-state process. 

The initial state presents the recommendation results according to semantic similarity to the search term and the processed metadata of the “Product” table. On the other hand, the recalculation state, assumes that the user has provided feedback by rating and/or discarding any recommendation results by flagging them as irrelevant. The provided feedback is used to predict the rating and relevance of the initially recommended items that were not evaluated by the user, thus refining the recommendation. If no feedback is provided, the recalculation state will produce the same recommendations as the initial.

The recommendation process is triggered when the user performs a new search with the desired search terms. The new search is logged to the “RecommenderSearch” table of the S4F database by the front-end and a unique RecommenderSearch ID is provided as input to the recommendation script. 

fashion_recommendation.py:
Triggers the fashion recommendation process.

#### Arguments:
```
* -i or -id (required): the RecommenderSearch ID
* -l or --loglevel (optional): the logging level
* -r or --recalc(optional): recalculation flag to trigger the recommendation after user’s feedback
```

#### Examples:
```python %PROJECT_HOME%/Recommender/fashion_recommendation.py -i 45 -l debug```

The Recommender retrieves the query information with id 45 from the RecommenderSearch table, expecting extensive event logging.

```python %PROJECT_HOME%/Recommender/fashion_recommendation.py -i 45 -l debug -r```

The recommendation engine refines the recommended items by providing an updated list which is taking into account the User’s feedback.

## 4. Science4Fashion Installation Guide:

1. Anaconda installation
	* download windows anaconda https://repo.anaconda.com/archive/Anaconda3-2020.11-Windows-x86_64.exe
	* install Anaconda with add to PATH option

2. Download editors 
	* notepad++ https://notepad-plus-plus.org/repository/7.x/7.0/npp.7.Installer.x64.exe
	* vscode https://code.visualstudio.com/download (only for developers)
	
3. Github desktop https://central.github.com/deployments/desktop/desktop/latest/win32 
	* Log in as i-Dare (only for developers)
	* clone [Science4Fashion_ref repository](https://github.com/i-Dare/Science4Fashion_ref/edit/main/README.md) to a dedicated path ex. `C:\Users\User\Documents\GitHub-Repos\Science4Fashion_ref`

4. Define enviroment variables:

	* Define PROJECT_HOME enviroment variable pointing to the cloned repository directory (`C:\Users\User\Documents\GitHub-Repos\Science4Fashion_ref`)
	* Define `PYTHONHOME` pointing to Anaconda installation
	* Define `PYTHONPATH` same as `PYTHONHOME`
	* Append `PROJECT_HOME` enviroment variable to `PYTHONPATH`
	
	Should be similar to these:
	* **PROJECT_HOME:** `C:\Users\User\Documents\GitHub-Repos\Science4Fashion_ref`
   	* **PYTHONHOME:** `C:\Users\User\anaconda3`   
	* **PYTHONPATH:** `C:\Users\User\anaconda3;%PROJECT_HOME%`
	
5. Install python packages:

	* ```$ pip install kmodes==0.10.2```
	* ```$ pip install scikit-fuzzy==0.4.2```
	* ```$ pip install textblob==0.15.3```
	* ```$ pip install plotly==4.14.3```
	* ```$ pip install prince==0.7.1```
	* ```$ pip install webcolors==1.11.1```
	* ```$ pip install light-famd```
	* ```$ conda install -c pytorch pytorch=1.7.1 torchvision=0.8.2```
	* ```$ pip install pymssql```
	* ```$ pip install opencv-python==4.5.1.48```
	* ```$ pip install tensorflow==2.4.1```
	* ```$ pip install fastai==1.0.61```
	* ```$ pip install wordsegment```
	* ```$ pip install py3-pinterest==1.2.2```
	* ```$ pip install instaloader==4.7.2```

---

**NOTE:** In case your [CPU does not support AVX or AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions), install `tensorflow` without AVX support by building it directly [from source](https://www.tensorflow.org/install/source_windows) or using prebuilt binaries for Windows from [tensorflow-windows-wheel](https://github.com/fo40225/tensorflow-windows-wheel) github repository.

---

	
6. Download NLTK content by executing the following commands in a cmd console:
	* for wordnet data:
	```$ python -c "import nltk;nltk.download('wordnet')"```
	* for stopwords:
	```$ python -c "import nltk;nltk.download('stopwords')"```
	* for part-of-speech tagger:
	```$ python -c "import nltk;nltk.download('averaged_perceptron_tagger')"```

	
7. Transfer AI models that reside in [Google Drive](https://drive.google.com/drive/folders/1OK_DCErAY8jta532aJRSX8ljZ8XnVqAJ?usp=sharing) to target system
	* Download latest models
	* Place models in %PROJECT_HOME%/resources/models/product_attributes (create directory if needed)
	
8. Edit DB connection details in %PROJECT_HOME%/config.json
	* Execute the following to make sure the database connection is responsive

	```$ python -c "import pandas as pd;import core.config as config;pd.read_sql_query('''SELECT Oid FROM %s.dbo.Product''' % config.DB_NAME, config.ENGINE)"```
	


	

	

