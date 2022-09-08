import pytest
import pickle
import sys
from lime_plot import Lime_interpret
sys.path.append("./")
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
dict_train=pload("2D_LSTM/dict_train.dict")
n_simulation,n_feature,sim_type,step_size,tot_time_frame=pload("2D_LSTM\sim_arguments.list")
raw_s2,dp_s2,p_10,data_s2,label_s2,projs=pload("2D_LSTM\data.list")


def test_a():
    print("------->test_2D_plot_non_int")
    from TwoD_plot import view_acc
    with pytest.raises(TypeError) as error:
       error=view_acc(dict_train,step_size,0.5) 
    assert str(error) == "<ExceptionInfo TypeError('The input is not int type. Please check the input.') tblen=3>"

def test_b():
    print("------->test_2D_plot_negative")
    from TwoD_plot import view_acc
    with pytest.raises(ValueError) as error:
       error=view_acc(dict_train,step_size,-1) 
    assert str(error) == "<ExceptionInfo ValueError('The number is less than 0. Please check the input.') tblen=3>"

def test_c():
    print("------->test_2D_lime_plot_negative")
    from lime_plot import Lime_interpret
    labels=label_s2
    projsT=projs.reshape(projs.shape[0],projs.shape[2],projs.shape[1])
    data=projsT
    with pytest.raises(ValueError) as error:
       error=exp=Lime_interpret(data,labels,-1)
    assert str(error) == "<ExceptionInfo ValueError('The number is less than 0. Please check the input.') tblen=3>"

def test_d():
    print("------->test_2D_lime_plot_non_int")
    labels=label_s2
    projsT=projs.reshape(projs.shape[0],projs.shape[2],projs.shape[1])
    data=projsT
    with pytest.raises(TypeError) as error:
       error=exp=Lime_interpret(data,labels,0.5)
    assert str(error) == "<ExceptionInfo TypeError('The input is not int type. Please check the input.') tblen=3>"


def test_e():
    print("------->test_1D_plot_non_int")
    n_simulations,step_size,tot_time_frame,n_features=pload("1D_LSTM/sim_argument.list")
    OneD_dict_train=pload("1D_LSTM/results.dict")
    from OneD_plot import view_acc
    with pytest.raises(TypeError) as error:
       error=view_acc(OneD_dict_train,step_size,0.5) 
    assert str(error) == "<ExceptionInfo TypeError('The input is not int type. Please check the input.') tblen=3>"

def test_f():
    print("------->test_1D_plot_negative")
    n_simulations,step_size,tot_time_frame,n_features=pload("1D_LSTM/sim_argument.list")
    OneD_dict_train=pload("1D_LSTM/results.dict")
    from OneD_plot import view_acc
    with pytest.raises(ValueError) as error:
       error=view_acc(OneD_dict_train,step_size,-1) 
    assert str(error) == "<ExceptionInfo ValueError('The number is less than 0. Please check the input.') tblen=3>"

if __name__ == '__main__':

    pytest.main("-s test_abcdef.py")
