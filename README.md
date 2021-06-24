# StockPrediction-
Streamlit dashboard
# Stock Market Live Update and Future Stock Value
Using this repository one can find live stock data of any company under `NSE S&P500` as well as can check what
will the future stock value for that particular company.

## Installation
1. Install [Python 3](https://www.python.org/) in your system. Make sure that `pip` is available for installing Python packages.
2. Install [`Conda`]
https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html for creating Conda Environments.
    ```bash
    pip install conda
    ```
3. Clone this repository or Extract it into a specific folder and `cd` into it.
    ```bash
    cd OrderFood-website
    ```
4. Create Conda environment called using `conda`.
    - Linux  or Mac
        ```bash
        conda create -n <environment name> python=<version>
        conda activate <environment name>
        ```
    - Windows
        ```
        conda create -n <environment name> python=<version>
        conda activate <environment name>
        ```
    You can use the `deactivate` command for deactivating the virtual environment.
5. Install the required Python packages by the command:
    ```bash
    pip install -r requirements.txt

    ```********Perfrom the solution part before installing through requirements.txt*********
###Solution for fbprophet installation error and twint library error
Fundamental step: Switch to your environment in your Anaconda prompt: conda activate name-of-your-python-enviornment

       Fundamental step: Switch to your environment in your Anaconda prompt: conda activate name-of-your-		  python-enviornment  

	Then the following steps shall work:

	On Prompt install Ephem:
	conda install -c anaconda ephem
	
	Install Pystan:
	conda install -c conda-forge pystan
	
	Finally install Fbprophet:
	conda install -c conda-forge fbprophet
#Twint
 	Enter the below command
	pip install --user --upgrade -e git+https://github.com/twintproject/twint.git@origin/master#egg=twint

#NLTK
	Nltk by default dosent install all the packages with pip, so while importing nltk enter nltk.download()         to install required packages:-
        import nltk
        nltk.download()

        You can remove the command after the packages are installed

 ## Usage 
1. You can start the streamlit server by calling
   ```bash
   streamlit run main.py
   ```
2. The website can be accessed through a browser at [127.0.0.1:8501](http://127.0.0.1:8501/) or [localhost:8501]
