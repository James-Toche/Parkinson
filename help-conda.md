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


# HELP WITH STALE STATE
conda install -n parkinson --force-reinstall python=3.14 libffi ipykernel
conda run -n parkinson python -c "import ctypes; print('ctypes OK')"
conda run -n parkinson python -m ipykernel install --user --name parkinson --display-name "Python (parkinson)" --force

# HELP WITH ML
- Logistic Regression uses straight lines to estimate classification probabilities simply and quickly.
- XGBClassifier combines a team of sequential decision trees to fix errors.
- SVC finds the widest possible margin between data groups for complex boundaries.
- Validation beats training accuracy due to strong regularization or an easier validation split.

# TMRW
- List 5 models that make sense to use in this dataset.
- The code we downloaded uses logistic regression, SVC, XGB; does this make sense to use, what if it's actually stupid?
- AI tells us to use Random Forest and others.
- We need to find a way to compare all these models, so which criteria should we use to determine the best model?