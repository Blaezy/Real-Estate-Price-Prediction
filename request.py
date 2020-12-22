# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 21:30:15 2020

@author: iamre
"""


import requests
url = 'http://localhost:5000/api'
r = requests.post(url,json={'exp':[1,2013,31.7,287.603,6,24.9804],})
