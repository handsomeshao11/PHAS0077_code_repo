import pickle
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../../")
from matplotlib import pyplot as plt
from utils import check_non_negative_integer
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

def view_acc(dict_train,step_size,time_edge):
    """
    Function to view the training accuracy

    :param data: The data generated by simulation
    :param step_size: The step size corresponding to the data
    :param time_edge: The end time you want to see
    """
    # check the input
    check_non_negative_integer(step_size)
    check_non_negative_integer(time_edge)
    # load data 
    steps_to_view=int(time_edge/step_size)
    x = [step_size*(i+1) for i in range (steps_to_view)]
    test=[ dict_train[v]["accuracy"][2][1]*100 for v in range (steps_to_view)]
    valid=[ max(dict_train[z]["accuracy"][1])*100 for z in range (steps_to_view)]
    train=[ max(dict_train[t]["accuracy"][0])*100 for t in range (steps_to_view)]

    # set the plot arguments 
    lsize=2
    msize=2
    mcolor="black"
    al=0.8

    # plot figure
    plt.figure(figsize=(15,8))
    plt.title("Accuracy VS. Time",fontdict={"family": "Times New Roman", "size": 25})
    plt.plot(x, test,color=(212/255,72/255,72/255), linewidth=lsize,marker="o",markersize=msize,markerfacecolor=mcolor,alpha=al,label="Test accuracy")
    plt.plot(x, train,"-" ,color=(0/255, 47/255, 167/255),linewidth=lsize,marker="o",markersize=msize,markerfacecolor=mcolor,alpha=al,label="Training accuracy")
    plt.plot(x, valid, "y-", linewidth=lsize,marker="o",markersize=msize,markerfacecolor=mcolor,alpha=al,label="Valid accuracy")
    plt.legend(prop={"family": "Times New Roman", "size": 20})
    plt.xticks(fontname="Times New Roman", fontsize=20)
    plt.yticks(fontname="Times New Roman", fontsize=20)
    plt.xlabel('Time',fontdict={"family": "Times New Roman", "size": 25})
    plt.ylabel('Accuracy',fontdict={"family": "Times New Roman", "size": 25})
    plt.savefig("2D_LSTM/s2_acc_{}".format(time_edge))

def top_n_important_features(dict_train,time,step_size,time_frame,top_number=1):
    """
    Function to get the location of the top n important features in the training

    :param dict_train: The result generated by simulation
    :param step_size: The step size corresponding to the data
    :param time: The end time you want to see
    :param top_number: The top n important features you want to see
    :return: A list of the location of the top n important features
    """
    # check the input
    check_non_negative_integer(step_size)
    check_non_negative_integer(time)
    # plot figure
    adrops = [] # the list to save all the mean adrop accuracy over time, by superimposing the accuracy of all previous time
    min_features_dict={}
    for i in range(int(time/step_size)):
        adrop = dict_train[i]["mltsa"] # get adrop data from dict_train, shape:(n_simulations,n_features)
        mean = np.mean(adrop, axis=0) # (72,) superposition of all simulations, shape(n_features)
        adrops.append(mean) # save the adroped features accuracy to the list by each time
    current_mean_adrops=np.mean(np.array(adrops),axis=0)# mean adrop array,
    no_rep_adrop=list(set(current_mean_adrops.tolist())) # eliminate elements of repetition: array->list->set->list

    if top_number>len(no_rep_adrop):
        raise ValueError("Your index is over the number of different accuracy of features, try smaller!") # check the index is not over the list length
    min_feature=np.min(no_rep_adrop)# find the minimum feature
    location=np.where(current_mean_adrops==min_feature)[0] # get the location of the minimum feature(s)

    if top_number > 1 : #  if the index is not the top 1 important adropped feature 
        for i in range(top_number-1):
            no_rep_adrop.remove(min_feature)# delete minimum feature(s)
            min_feature=np.min(no_rep_adrop)# find minimum feature(s) after deleting the last minimum feature(s)
            location=np.where(current_mean_adrops==min_feature)[0]# find the loation of the minimum feature(s)

    return location

def most_important_features(dict_train):
    """
    Function to get the location of the most important feature(s)

    :param dict_train: The result generated by simulation
    :return: The locations of the most important feature(s)
    """
    adrops = []
    min_dict={}
    num=0
    for run in range(len(dict_train)):
    #     plt.figure()
        adrop = dict_train[run]["mltsa"] # (200,72)
    #     print(np.array(adrop).shape) # (200,72)
        mean = np.mean(adrop, axis=0) # (72,)
        adrops.append(mean)
        time_mean_adrops=np.mean(np.array(adrops),axis=0)# (73,)
    #     print(time_mean_adrops.shape)
        min_feature=np.min(time_mean_adrops)
        location=np.where(time_mean_adrops==min_feature)[0]
        min_dict[num]=location
        num+=1
    return min_dict
# min_dict=most_important_features(dict_train)


def dots_plot_most_important_features(dict_train):
    """
    Function to plot the most important feature(s) at each time point

    :param dict_train: The result generated by simulation
    """
    min_dict=most_important_features(dict_train)
    plt.figure(figsize=(15,10))
    plt.title("Location of the Worst Accuracy Droped Features VS. Time",x=0.5,y=1)
    for time in range(len(min_dict)):
        plt.scatter([step_size*list(min_dict.keys())[time]+step_size]*min_dict[time].shape[0],min_dict[time])
    plt.ylabel("Feature Location")
    plt.xlabel("Time")
    plt.savefig("2D_LSTM/dots_plot_most_important_features.png")

def draw_top_1_important_feature(dict_train,time,step_size,time_frame,top_number=1):
    """
    Function to plot the top 1 important feature(s)

    :param dict_train: The result generated by simulation
    :param time: The running time step you want 
    :param step_size: The step size corresponding to the data
    :param top_number: The top n important features you want to see
    """
    # check the input
    check_non_negative_integer(step_size)
    check_non_negative_integer(time)
    # plot figure
    min_feature_list=top_n_important_features(dict_train,time,step_size,time_frame,top_number)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(10, 10))
    plt.title("Adropped Feature of  Min Accuracy in {}".format(time))
    for i in range(len(p_10.coeff)):
        if i in min_feature_list:
            theta = p_10.coeff[i]
            temp = [np.cos(theta) * 2, np.sin(theta) * 2]
            ax.plot([0,theta],[0,1/top_number],label="top {}".format(top_number),color="black",linewidth=3,marker='o',markersize=5)# theta = rads, 5= line length
            ax.bar(theta,1/top_number, width=np.pi/15,color="yellow",alpha=0.5)
            ax.set_rlabel_position(-20)
    ax.legend(bbox_to_anchor=(1, 1.1))
    plt.savefig("2D_LSTM/2D_top_1_feature_{}.png".format(time))

def draw_n_important_features(dict_train,time,step_size,time_frame,top_number=1):
    """
    Function to plot the top n important features

    :param dict_train: The result generated by simulation
    :param time: The running time step you want 
    :param step_size: The step size corresponding to the data
    :param top_number: The top n important features you want to see
    """
    # check the input
    check_non_negative_integer(step_size)
    check_non_negative_integer(time)
    # plot figure
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(10, 10))
    for top in range(top_number):
        min_feature_list=top_n_important_features(dict_train,time,step_size,time_frame,top+1)
        plt.title("Features in {}".format(time),fontdict={"family": "Times New Roman", "size": 35})
        for i in range(len(p_10.coeff)):
            if i in min_feature_list:
                theta = p_10.coeff[i]
                temp = [np.cos(theta) * 2, np.sin(theta) * 2]
                ax.plot([0,theta],[0,1/(top+1)],label="top {}".format(top+1),linewidth=4,marker='o',markersize=5)# theta = rads, 5= line length
                ax.bar(theta,1/(top+1), width=np.pi/15,color="grey",alpha=0.4)
                ax.set_rlabel_position(-20)
        ax.legend(bbox_to_anchor=(1, 1.1))
    plt.legend(prop={"family": "Times New Roman", "size": 20},loc='upper left', bbox_to_anchor=(0.92,1))
    plt.savefig("2D_LSTM/2D_features_{}.png".format(time))

if __name__ == '__main__':
    # load data
    raw_s2,dp_s2,p_10,data_s2,label_s2,projs=pload("2D_LSTM/data.list")
    dict_train=pload("2D_LSTM/dict_train.dict")
    n_simlations,n_features,data_shape,step_size,tot_time_frame=pload("2D_LSTM/sim_arguments.list")
    n_step=tot_time_frame/step_size
    # reproduce the features 
    plt.figure(figsize=(12,8))
    p_10.show_axis()
    plt.show()

    # plot the figures, you can change the time to see different plot. Remember the time should be included in data.list which you have generated.
    view_acc(dict_train,step_size,500) 
    dots_plot_most_important_features(dict_train)
    draw_top_1_important_feature(dict_train,200,step_size,tot_time_frame,1)
    draw_n_important_features(dict_train,200,step_size,tot_time_frame,3)
    print("--plots have been generated--")