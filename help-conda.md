## HELP WITH ENVIRONMENTS

This assumes you are using conda (miniconda)


# 1. Create an environment (named env_name) for Python

$ conda create --name env_name python --channel conda-forge


2. Activate the environment

$ conda activate env_name


3. Do stuff like

$ conda install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn tqdm

$ conda update conda

$ conda update python

$ conda update --all


## HELP WITH JUPYTER NOTEBOOKs

1. Install required packages

$ conda install jupyterlab ipykernel ipywidgets


2. Configure notebook settings

$ jupyter notebook --generate-config

Go to .jupyter folder and open the jupyter_notebook_config.py 

Add the following to the end of the file:

# Comment added by James Toche | for Windows
import webbrowser
webbrowser.register('edge', None, webbrowser.BackgroundBrowser('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'))
c.NotebookApp.browser = 'edge'

# Comment added by Patrick Toche | for MacOS
import webbrowser
webbrowser.register('safari', None, webbrowser.BackgroundBrowser('/Applications/Safari.app/Contents/MacOS/Safari'))
c.NotebookApp.browser = 'safari'


3. Register the preferred kernel
$ jupyter kernelspec list 
$ python -m ipykernel install --user --name=env_name --display-name="Python (env_name)"


# HELP WITH WINDOWS POWERSHELL
1. Run as administrator
2. Type 
$ Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

3. Type
$ conda init powershell