import sys
import findspark
import pandas as pd
import os
findspark.init()

from pyspark.sql import SQLContext
from pyspark import SparkContext 
from pyspark.sql import SparkSession
from pyspark.sql import functions

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from nltk.corpus import stopwords

import numpy as np

print("hello kuro program spark")
print (sys.argv[0])
spark = SparkSession.builder\
                    .appName('PhanMinhHung_Thesis')\
                    .master("local[*]")\
                    .config("spark.executor.memory", "12gb")\
                    .getOrCreate()
print(spark)
#Prepare Data
dataset = spark.read.csv(r'dataset.csv', header = True, inferSchema = True)
#dataset = spark.read.csv(r'C:\Users\Admin\100ktrainingset.csv', header = True, inferSchema = True)
drop_list = ['_c0']
dataset = dataset.select([column for column in dataset.columns if column not in drop_list])
dataset = dataset.na.drop()
yes = dataset.filter(functions.col('Should-Read')== 1)
no = dataset.filter(functions.col('Should-Read')== 0)
dataset = yes.unionAll(no)
dataset.printSchema()
#split test data
testdata = dataset.sample(fraction = 0.3, seed = 100)
testdata.toPandas().to_csv(r'\testdata.csv')
# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="Book-Title", outputCol="words", pattern="\\W")
# stop words
add_stopwords = stopwords.words('english')
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
# bag of words count
countVectors = CountVectorizer(inputCol="words", outputCol="features",vocabSize=10000, minDF=100)
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors])
# Fit the pipeline to training documents.
pipelineFit = pipeline.fit(dataset)
#save Pipeline model 
pipelineFit.write().overwrite().save(r'\pipeline')
#trainingData = pipelineFit.transform(dataset)
print('done')
spark.stop()