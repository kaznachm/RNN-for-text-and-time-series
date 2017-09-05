#!/bin/sh
sudo apt-get update
sudo apt-get install htop
sudo apt-get install python-pip
export LC_ALL=C
sudo pip install numpy
sudo pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt
sudo pip install Lasagne==0.1
sudo pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
sudo pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
sudo pip install Theano==0.8.2