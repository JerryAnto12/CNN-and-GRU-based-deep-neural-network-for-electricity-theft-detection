from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow
from keras.layers import Conv1D, MaxPooling1D, GRU, Flatten, Dense
from keras.models import Sequential
from tensorflow.keras.layers import Dropout

app = Flask(__name__)

# Load the entire model
model = tf.keras.models.load_model('model-F3-M1-64')

