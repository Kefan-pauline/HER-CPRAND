# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:58:01 2021

@author: sunke
"""
from src._image import image_boxplot
import numpy as np

def test_image_boxplot():
    r=15
    n_samples = int(10*r*np.log(r)+1)
    n_samples_err=400
    image_boxplot(r,n_samples,n_samples_err,exact_err=False,tol=0.08)