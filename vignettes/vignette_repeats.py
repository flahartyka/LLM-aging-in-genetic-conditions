import pandas as pd
import ast 
import openai
import tiktoken
from scipy import spatial 
from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import seaborn as sns
from tensorflow.python.ops.numpy_ops import np_config 
from sentence_transformers import SentenceTransformer
np_config.enable_numpy_behavior()


#load embeddings model
model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

#embedding definition
def embed(input):
  return model.encode(input)

#gets correlation between two embeddings
def get_corr(first, second): 
    corr = np.inner(first, second)
    return corr

vignette_file = open("vignette_repeats.txt", "r")
data = vignette_file.read()
vignette_list = data.split('\n')

#input the original vignette here
original = ["A 35-year-old woman comes to the clinic complaining of shortness of breath and fatigue. She has a history of recurrent respiratory infections and was born with polydactyly (an extra finger on each hand). Physical examination reveals a small head size, prominent forehead, widely spaced eyes, short nose, and a thin upper lip. The patient has a soft, high-pitched voice and a small jaw. She also has a history of recurrent urinary tract infections and kidney stones. Abdominal examination reveals an enlarged liver and kidneys. Imaging studies show holoprosencephaly (a single, large brain hemisphere), hydrocephalus (fluid accumulation in the brain), and Dandy-Walker syndrome (a congenital brain malformation). The patient is diagnosed with Meckel syndrome, a rare genetic disorder that affects the development of multiple organs and systems."]

q1 = embed(original)

get_corr(q1, q2)

corr_list = []
for vignette in vignette_list:
    q2 = embed(vignette)
    num = get_corr(q1, q2)
    corr_list.append(num[0])


corr_list.to_csv('correlation_vals.csv')
