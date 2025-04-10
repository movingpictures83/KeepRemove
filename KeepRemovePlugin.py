import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import argparse
import warnings
from sklearn.metrics import roc_curve, auc
warnings.filterwarnings('ignore')

import PyIO
import PyPluMA

class KeepRemovePlugin:
 def input(self, inputfile):
  self.parameters = PyIO.readParameters(inputfile)
 def run(self):
  pass
 def output(self, outputfile):
  model = XGBClassifier()
  model.load_model(PyPluMA.prefix()+"/"+self.parameters["model"])
  stats = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["stats"], sep="\t")

  # Print SIDR predictions to stats dataframe
  stats['SIDR_predictions'] = model.predict(stats.drop(['contig', 'TopHit', 'MostHits', 'Origin'], axis=1))

  # Save the stats dataframe to a new tsv file
  stats.to_csv(outputfile+".full.predictions.tsv", sep='\t', index=False)

  # Save the stats dataframe to a new tsv file if the SIDR_predictions column equals 1
  stats[stats['SIDR_predictions'] == 1].to_csv(outputfile+".keep.predictions.tsv", sep='\t', index=False)
  stats[stats['SIDR_predictions'] == 0].to_csv(outputfile+".remove.predictions.tsv", sep='\t', index=False)

