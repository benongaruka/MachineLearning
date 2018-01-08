# -*- coding: utf-8 -*-
"""
Created on Sun Jan 07 22:32:22 2018
#General python programming code snippets
@author: Benon
"""
# *args and **kwargs: magic parameters for when the number of arguments is not known before hand, *args is for non key worded arguements, **kwargs for keyworded arguments

# *args
def test_var_args(f_arg, *argv):
    print("first normal arg:", f_arg)
    for arg in argv:
        print("another arg through *argv:", arg)
test_var_args("Benon", "test", "we","walks")

def greet_me(**kwargs):
    for key, value in kwargs.items():
        print("{0} = {1}".format(key,value))
        
greet_me(name="Benon")

#You can also call functions using *argv and *kwargs
def test_args_kwargs(arg1, arg2, arg3):
    print("arg1:", arg1)
    print("arg2:", arg2)
    print("arg3:", arg3)

#With *args
args = ("two",3,5)
test_args_kwargs(*args)

#With **kwargs
kwargs = {"arg3": 3, "arg2":"two", "arg1":5}
test_args_kwargs(**kwargs)
