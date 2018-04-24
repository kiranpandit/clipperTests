##kiran computer workaround
import sys
sys.path.insert(0,'/Library/Python/2.7/site-packages/')
import numpy
import pandas as pd
import pyspark
from pyspark.ml.feature import *
from pyspark.sql import SparkSession


spark = SparkSession                .builder                .appName("adult-data")                .getOrCreate()

cols = ['age','workclass','fnlwgt','education','education-num','marital-status', \
	'occupation','relationship','race','sex','capital-gain', \
	'capital-loss','hours-per-week','native-country','class']

data = spark.createDataFrame(pd.read_csv('adult.csv', header=None, names=cols))

wor_ind = pyspark.ml.feature.StringIndexer(inputCol='workclass', outputCol='wor_label').fit(data)
data = wor_ind.transform(data)

edu_ind = pyspark.ml.feature.StringIndexer(inputCol='education', outputCol='edu_label').fit(data)
data = edu_ind.transform(data)

mar_ind = pyspark.ml.feature.StringIndexer(inputCol='marital-status', outputCol='mar_label').fit(data)
data = mar_ind.transform(data)

occ_ind = pyspark.ml.feature.StringIndexer(inputCol='occupation', outputCol='occ_label').fit(data)
data = occ_ind.transform(data)

rel_ind = pyspark.ml.feature.StringIndexer(inputCol='relationship', outputCol='rel_label').fit(data)
data = rel_ind.transform(data)

rac_ind = pyspark.ml.feature.StringIndexer(inputCol='race', outputCol='rac_label').fit(data)
data = rac_ind.transform(data)

sex_ind = pyspark.ml.feature.StringIndexer(inputCol='sex', outputCol='sex_label').fit(data)
data = sex_ind.transform(data)

nat_ind = pyspark.ml.feature.StringIndexer(inputCol='native-country', outputCol='nat_label').fit(data)
data = nat_ind.transform(data)

cla_ind = pyspark.ml.feature.StringIndexer(inputCol='class', outputCol='cla_label').fit(data)
data = cla_ind.transform(data)

data = data.select(['age','wor_label','fnlwgt','edu_label','education-num','mar_label', \
	'occ_label','rel_label','rac_label','sex_label','capital-gain', \
	'capital-loss','hours-per-week','nat_label','cla_label'])

data.write.csv('adultLabel.csv')
