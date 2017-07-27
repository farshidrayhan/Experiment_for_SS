#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 04:47:39 2017

@author: farshid
"""

from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')

utils.chooseCRANmirror(ind=1)

packnames = ('rcdk')
utils.install_packages(StrVector(packnames))

rcdk = rpackages.importr('rcdk')

d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
thatpackage = importr('rcdk', robject_translations = d)






#import urllib2
from urllib.request import urlopen
import urllib.request
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

bioc_url = urllib.request.urlopen("https://raw.github.com/hadley/stringr/master/R/c.r")
string = ''.join(bioc_url.readlines())

stringr_c = SignatureTranslatedAnonymousPackage(string, "stringr_c")



import urllib.request

wp = urllib.request.urlopen("https://raw.github.com/hadley/stringr/master/R/c.r")
pw = ''.join( str( wp.readlines() ) )
print(pw)

stringr_c = SignatureTranslatedAnonymousPackage(pw, "pw")










