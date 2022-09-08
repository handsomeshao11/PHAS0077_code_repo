# PHAS0077_code_repo
This is the PHAS0077 code repository of Shao Juexi.

Please install the following packages first:
mdtraj
numpy
scipy
pandas
seaborn
tensorflow (2.6 and GPU perfered)
scikit-learn
matplotlib
jupyter notebook / jupyterlab
lime

To run the generated data, please download the data from [PHAS0077_support_data](https://www.dropbox.com/scl/fo/2or413mn5m7e7lqmoxipp/h?dl=0&rlkey=6fzfyihixvqq2po4zk0e2khnq).

# 1D
## data generating and training
Run 1D Machine learning program in the terminal with:

`python .\1D_script.py`
## plot
After you finished the training, you can create a new folder, and move the files generated and the 1D_plot.py script into the folder, and running by:

`python .\1D_plot.py`

# 2D data generator
## data generating and training
You can run 2D data machine learning program here:
Run 2D Machine learning program in the terminal with:

`python .\2D_script.py`
## plot
After you finished the training, you can create a new folder, and move the files generated and the 2D_plot.py script into the folder, and running by:

`python .\2D_plot.py`
## LIME 
Furthermore:
If you want to see the LIME results for 2D, you move the lime.py to the folder, and running by:

`python .\lime.py`

And there are some lime instances in the 2D_LSTM folder, you can use pload function(see it in each python file) to see the result directly.

The files mentioned above was finished by Shao Juexi personally.
