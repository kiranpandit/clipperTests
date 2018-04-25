##kiran computer workaround
import sys
sys.path.insert(0,'/Library/Python/2.7/site-packages/')
import numpy

from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers.pyspark import deploy_pyspark_model

import pandas as pd
import pyspark
import os
import urllib
import sys
from StringIO import StringIO

from pyspark.sql.functions import *
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *
from pyspark.sql import SparkSession


spark = SparkSession                .builder                .appName("clipper-pyspark")                .getOrCreate()
sc = spark.sparkContext

clipper_conn = ClipperConnection(DockerContainerManager())
clipper_conn.start_clipper()

data = spark.createDataFrame(pd.read_csv('iris.csv', header=None, names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']))

# vectorize all numerical columns into a single feature column
feature_cols = data.columns[:-1]
assembler = pyspark.ml.feature.VectorAssembler(inputCols=feature_cols, outputCol='features')
data = assembler.transform(data)

# convert text labels into indices
data = data.select(['features', 'class'])
label_indexer = pyspark.ml.feature.StringIndexer(inputCol='class', outputCol='label').fit(data)
data = label_indexer.transform(data)

# only select the features and label column
data = data.select(['features', 'label'])

reg = 0.01
train, test = data.randomSplit([0.70, 0.30])
lr = pyspark.ml.classification.LogisticRegression(regParam=reg)
model = lr.fit(train)

def predict(spark, model, inputs):
	from pandas import read_csv
	TESTDATA = StringIO(inputs[0])
	data = spark.createDataFrame(read_csv(TESTDATA, header=None, names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']))
	feature_cols = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
	assembler = pyspark.ml.feature.VectorAssembler(inputCols=feature_cols, outputCol='features')
	data = assembler.transform(data)
	data = data.select(['features', 'class'])
	label_indexer = pyspark.ml.feature.StringIndexer(inputCol='class', outputCol='label').fit(data)
	data = label_indexer.transform(data)
	data = data.select(['features', 'label'])
	output = model.transform(data).select("prediction").rdd.flatMap(lambda x: x).collect()
	return output

deploy_pyspark_model(
    clipper_conn,
    name="iris-output",
    version=1,
    input_type="string",
    func=predict,
    pyspark_model=model,
    sc=sc,
    pkgs_to_install=["pandas"])

clipper_conn.register_application(
	name="iris-app",
	input_type="strings",
	default_output="-1",
    slo_micros=9000000) #will return default value in 9 seconds

clipper_conn.link_model_to_app(app_name="iris-app", model_name="iris-output")

