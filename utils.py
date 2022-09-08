def check_non_negative_integer(number):
    if type(number) != int:
        raise TypeError("The input is not int type. Please check the input.") 
    elif number<0:
        raise ValueError("The number is less than 0. Please check the input.")
