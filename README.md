# Science4Fashion Refactoring Repository

## Introduction
The Science4Fashion project aims at facilitating the design of clothing products, particularly in the field of product concept development, by providing personalized proposals to the designer as inspiration, by developing tool based on Artificial Intelligence methods. The tools will work in addition to existing support systems for the development of clothing products such as Life Cycle Management Systems, CAD (Graphics), 3D Modeling systems, commonly used by companies and designers.

The System is comprised of the following components:
1. [Data Collection](data_collection)
2. [Data Annotation](image_annotation)
3. [Clustering](clustering)
4. [Clothing Recommender with User Feedback Component](recommender)

## [1. Data Collection](#data_collection)
For data collection the system targets a number of well known online resources and collects images, text and metadata according to a user defined query. There data sources can be grouped to website-based, that are accesed via a number of web crawlers that collect the nessecery information and social media-based that usually contain trending content that is rich but noisy. The data sources are refered as `adapters`, and each adapter is responsible for a single data source. Namely, the web-based adapters are Asos, sOliver, and Zalando, whereas the social media-based are Instagram and Pinterest.

The data collection process is initiated by the user who provides a search term and a valid adaptor invoking the `WebCrawlersWrapper.py` script. The script is a wrapper of all the implemented adapters and takes as arguments the name of the adapter, a search term and optionally the maximum number of results. The search term should contain keywords that would otherwise be used as a query to a fashion retailers eshop or to a search engine and they should best describe the desired content. Afterwards, the selected adapter will query the respective source for the desired content and attempt to capture a number of attributes for each result. The product name, brand, image, description, genre, price and other metadata will be stored in the system's database. 

As soon as the adapter finishes retrieving the requested information from the source, the automated annotation process is initiated that will seek to infer specific clothing attributes regarding the color, neck-line, sleeves, length and fit of the clothing article. The annotation is driven by the retrieved image, the description and the metadata.

## [2. Image Annotation](#image_annotation)
The image annotation process is executed at the end of the adapter and is responsible for extracting the main product attributes. At this time, the system will attempt to capture the five most dominant colors of the product. Firstly, the product image is processed and the background is extracted. Secondly, the algorith discards any parts of the foreground that may contain skin color information, thus creating a mask for the clothing article. Finally, the color information inside the mask will be decomposed to the 5 most dominant colors and each color will be stored in the database along with the product ID.

Apart from the color information, the rest of the product's attributes regarding the collar style, length, neckline, sleeve type and general fit of the article are infered through a pretrained Deep Neural Network. The DNN is trained on on popular fashion datasets and employs adopted architectures such as VGG16, ResNet50 and ResNet50v2. 

## [3. Clustering](#clustering)
## [4. Clothing Recommender with User Feedback](#recommender)


## Science4Fashion Installation Guide:

1. Anaconda installation
	* download windows anaconda https://repo.anaconda.com/archive/Anaconda3-2020.11-Windows-x86_64.exe
	* install Anaconda with add to PATH option
	
2. Github desktop https://central.github.com/deployments/desktop/desktop/latest/win32
	* Log in as i-Dare 
	* clone Science4Fashion_ref repository (C:\Users\User\Documents\GitHub-Repos\Science4Fashion_ref)
	
3. Download editors
	* notepad++ https://notepad-plus-plus.org/repository/7.x/7.0/npp.7.Installer.x64.exe
	* vscode https://code.visualstudio.com/download

4. Define enviroment variables:

	* Define PROJECT_HOME enviroment variable pointing to the cloned repository directory (`C:\Users\User\Documents\GitHub-Repos\Science4Fashion_ref`)
	* Define `PYTHONHOME` pointing to Anaconda installation
	* Define `PYTHONPATH` same as `PYTHONHOME`
	* Append `PROJECT_HOME` enviroment variable to `PYTHONPATH`
	
	Should be similar to this:
	* **PROJECT_HOME:** `C:\Users\User\Documents\GitHub-Repos\Science4Fashion_ref`
   	* **PYTHONHOME:** `C:\Users\User\anaconda3`   
	* **PYTHONPATH:** `C:\Users\User\anaconda3;%PROJECT_HOME%`
	
5. Install python packages:
	* ```$ pip install kmodes==0.10.2```
	* ```$ pip install skfuzzy==0.4.2```
	* ```$ pip install textblob==0.15.3```
	* ```$ pip install instaloader==4.5.5```
	* ```$ pip install plotly==4.14.3```
	* ```$ pip install prince==0.7.1```
	* ```$ pip install webcolors==1.11.1```
	* ```$ pip install light-famd```
	* ```$ conda install -c pytorch torchvision```
	* ```$ pip install pymssql```
	* ```$ pip install opencv-python==4.0.1```
	* ```$ pip install --upgrade tensorflow```
	* ```$ pip install fastai```
	* ```$ pip install wordsegment```
	* ```$ pip install py3-pinterest```
	* ```$ pip install light-famd```

	
6. Download NLTK content by executing the following commands in a cmd console:
	* for stopwords:
	```$ python -c "import nltk;nltk.download('stopwords')"```
	* for part-of-speech tagger:
	```$ python -c "import nltk;nltk.download('averaged_perceptron_tagger')"```

	
7. Transfer AI models that reside in [Google Drive](https://drive.google.com/drive/folders/1OK_DCErAY8jta532aJRSX8ljZ8XnVqAJ?usp=sharing) to target system
	* Download latest models
	* Place models in %PROJECT_HOME%/resources/models (create directory if needed)
	
8. Edit DB connection details in %PROJECT_HOME%/config.json
	* Execute the following to make sure the database connection is responsive

	```$ python -c "import pandas as pd;import helper_functions;pd.read_sql_query('''SELECT * FROM %s.dbo.Product''' % helper_functions.DB_NAME, helper_functions.ENGINE)"```
	


	

	

