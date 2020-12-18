# Science4Fashion Refactoring Repository

## Project installation guide:

1. Anaconda installation
	* download windows anaconda https://repo.anaconda.com/archive/Anaconda3-2020.11-Windows-x86_64.exe
	* install Anaconda with add to PATH option
	
2. Github desktop https://central.github.com/deployments/desktop/desktop/latest/win32
	* Log in as i-Dare 
	* clone Science4Fashion_ref repository (C:\Users\User\Documents\GitHub-Repos\Science4Fashion_ref)
	
3. Download editors
	* notepad++ https://notepad-plus-plus.org/repository/7.x/7.0/npp.7.Installer.x64.exe
	* vscode https://code.visualstudio.com/download

5. Define enviroment variables:

	* Define PROJECT_HOME enviroment variable pointing to the cloned repository directory (`C:\Users\User\Documents\GitHub-Repos\Science4Fashion_ref`)
	* Define `PYTHONHOME` pointing to Anaconda installation
	* Define `PYTHONPATH` same as `PYTHONHOME`
	* Append `PROJECT_HOME` enviroment variable to `PYTHONPATH`
	
	Should be similar to this:
	* **PROJECT_HOME:** `C:\Users\User\Documents\GitHub-Repos\Science4Fashion_ref`
   	* **PYTHONHOME:** `C:\Users\User\anaconda3`   
	* **PYTHONPATH:** `C:\Users\User\anaconda3;%PROJECT_HOME%`
	
6. Install python packages:
	* ```$ pip install kmodes==0.10.2```
	* ```$ pip install skfuzzy==0.4.2```
	* ```$ pip install textblob==0.15.3```
	* ```$ pip install instaloader==4.5.5```
	* ```$ pip install instalooter==2.4.4```
	* ```$ pip install prince==0.7.1```
	* ```$ pip install webcolors==1.11.1```
	* ```$ conda install -c pytorch torchvision```
	* ```$ pip install pymssql```
	* ```$ conda install -c conda-forge opencv```
	* ```$ pip install --upgrade tensorflow```
	* ```$ pip install fastai```
	* ```$ pip install wordsegment```
	* ```$ pip install py3-pinterest```
	
	
7. Download NLTK stopwords by, opening a cmd console and executing:
	```$ python -c "import nltk;nltk.download('stopwords')"```

	
8. Transfer AI models to target system https://drive.google.com/drive/folders/1OK_DCErAY8jta532aJRSX8ljZ8XnVqAJ?usp=sharing
	* Download latest models
	* Place models in %PROJECT_HOME%/ImageAnnotation/Prediction/models directory (create if needed)
	
9. Edit DB connection details in %PROJECT_HOME%/config.json
	* Execute the following to make sure the database connection is responsive

	```$ python -c "import pandas as pd;import helper_functions;pd.read_sql_query('''SELECT * FROM %s.dbo.Product''' % helper_functions.DB_NAME, helper_functions.ENGINE)"```
	


	

	

