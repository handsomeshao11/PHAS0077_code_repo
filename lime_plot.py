from lime.lime_tabular import RecurrentTabularExplainer
from utils import check_non_negative_integer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
import numpy as np
import time
import sys
sys.path.append("../../")
sys.path.append("../")
from matplotlib import pyplot as plt
def Lime_interpret(data,labels,end_time):
    """
    Function to do LIME interpret for the model

    :param data: The data genertated from the training
    :param labels: The labels of the data
    """
    # check the input
    check_non_negative_integer(end_time)
    # interpret
    start=time.time()
    time_frame = [0, end_time]
    X_train, X_test, y_train, y_test = train_test_split(data[:,time_frame[0]:time_frame[1],:], labels, test_size=0.3, random_state=42)

    feature_name=[]
    name= [i for i in range(n_feature) ]
    for i in name:
        feature_name.append(str(i+1))
        
    explainer = RecurrentTabularExplainer(X_train,training_labels=y_train,feature_names=feature_name)
    model = tf.keras.models.load_model("2D_LSTM\2D_LSTM_0to{}.model".format(end_time))
    exp = explainer.explain_instance(X_test[0], model.predict)
    over=time.time()
    print("It take {} minutes to interpret.".format((over-start)/60))
    return exp

def psave(filename, object):
    """
    Function to save files

    :param filename: The name string you want to save 
    :param object: The object you want to save
    """
    file_write=open(filename, "wb")
    pickle.dump(object, file_write)
    file_write.close()
    return
def pload(filename):
    """
    Function to load files

    :param filename: The name string you want to load
    :return: The file
    """
    file_read=open(filename,mode="rb")
    file = pickle.load(file_read)
    file_read.close()
    return file

if __name__ == '__main__':
    # load data
    raw_s2,dp_s2,p_10,data_s2,label_s2,projs=pload("2D_LSTM\data.list")
    labels=label_s2
    n_simulation,n_feature,sim_type,step_size,tot_time_frame=pload("2D_LSTM\sim_arguments.list")

    projsT=projs.reshape(projs.shape[0],projs.shape[2],projs.shape[1])
    data=projsT

    exp=pload("2D_LSTM\lime_140.instance") # you can just load a instance here 
    # exp=Lime_interpret(data,labels,140) # or you can interept a new data here
    fig = exp.as_pyplot_figure() # show the result plot
    exp.show_in_notebook(show_table=True, show_all=False)# show the result plot in jupyter notebook
    fig.savefig('2D_LSTM\lime140.png')
    psave("2D_LSTM\lime_140.instance",exp)
    print("--LIME plot saved--")