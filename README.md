# PHAS0077_code_repo
This is the PHAS0077 code repository of Shao Juexi.

Please create a new environment and install the following packages:
numpy
, scipy
, pandas
, seaborn
, tensorflow (2.6 and GPU perfered)
, scikit-learn
, matplotlib
, jupyter notebook / jupyterlab
, lime
, pytest
, MLTSA.
Or you can use `pip install -r requirements.txt` to install all the packages.

**Because the plot programs have fixed relative path both in 1D and 2D, after generating the new data both in 1D and 2D, you need to remove the past data and put the new data into the corresponding folder.**

Because of the big memory of the data, to plot results with past data, you could download the data from [PHAS0077_support_data](https://www.dropbox.com/scl/fo/2or413mn5m7e7lqmoxipp/h?dl=0&rlkey=6fzfyihixvqq2po4zk0e2khnq). The each folder corresponds to the folder in github, and simply put each data into the correspond folder(LSTM and MLP used same data).

# 1D
## plot
You could use the current data to plot the results. Run with:

`python .\OneD_plot.py`

## data generating and training
In the OneD_script file, you can choose to using the current data, and train the model. Make sure you put the 1D_data.list and sim_arguments.list in the  PHAS0077_code_repo folder. Or you could generate new 1D data and run 1D Machine learning program. The instruction of choice was commented in the code.
Then you can run with

`python .\OneD_script.py`

After you finished the training, you can move the files generated into the 1D_LSTM or 1D_MLP folder(if you use MLP, remember to change all the path to 1D_MLP), and run with:
`python .\OneD_plot.py`

# 2D
## plot
You could use the current data to plot the results and run with:

`python .\TwoD_plot.py`

## LIME
There are some lime instances in the 2D_LSTM folder, if you want to see the LIME results for 2D,  you can run with:

`python .\lime_plot.py`

Or you can change the lime_plot file to interpret a new instance, the instructions of changing was commented in the file. The Lime program is only for LSTM currently.
## data generating and training
In the TwoD_script file, you can choose to using the current data, and train the model. Make sure you put the data.list in the  PHAS0077_code_repo folder. Or you could generate new 2D data and run 2D Machine learning program. The instruction of choice was commented in the code.
Then you can run with

`python .\TwoD_script.py`.

After you finished the training, you can move the files generated into the 2D_LSTM and 2D_MLP (if you use MLP, remember to change all the path to 2D_MLP)folder, and run by:
`python .\TwoD_plot.py`

# Test
The utils file contains the functions to throw Exceptions.
The test file is used to check the abnormal input. You can run it with:

`pytest .\test.py`
#
The files mentioned above was finished by Shao Juexi personally.
