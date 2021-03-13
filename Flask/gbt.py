import sys
import findspark
import pandas as pd
import os
findspark.init()

from pyspark.sql import SQLContext
from pyspark import SparkContext 
from pyspark.sql import SparkSession
from pyspark.sql import functions
from pyspark.sql.functions import col

#from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from nltk.corpus import stopwords
from pyspark.ml import PipelineModel
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import GBTClassificationModel
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support,accuracy_score,f1_score,recall_score,precision_score

def trainGBT(maxIter, maxDepth,stepSize):
	spark = SparkSession.builder\
                    .appName('PhanMinhHung_Thesis')\
                    .master("local[*]")\
                    .config("spark.executor.memory", "12gb")\
                    .getOrCreate()
	print(spark)
	dataset = spark.read.csv(r'\dataset.csv', header = True, inferSchema = True)
	dataset = dataset.withColumn('Book-Rating', dataset["Book-Rating"].cast("double"))
	dataset = dataset.withColumn('Should-Read', dataset["Should-Read"].cast("double"))
	drop_list = ['_c0']
	dataset = dataset.select([column for column in dataset.columns if column not in drop_list])
	dataset = dataset.na.drop()
	yes = dataset.filter(functions.col('Should-Read')== 1)
	no = dataset.filter(functions.col('Should-Read')== 0)
	dataset = yes.unionAll(no)
	dataset.printSchema()
	pipelineFit = PipelineModel.load(r'\pipeline')
	trainingData = pipelineFit.transform(dataset)
	#GBT
	gbt = GBTClassifier(labelCol="Should-Read", featuresCol="features")
	gbt.setMaxIter(maxIter)
	gbt.setMaxDepth(maxDepth)
	gbt.setStepSize(stepSize)
	print(gbt)
	gbtModel = gbt.fit(trainingData)
	gbtModel.write().overwrite().save('/gbtm')
	print(gbtModel)
	model = gbtModel.toDebugString
	print('done training')
	spark.stop()
	return model
def testGBT(link):
	spark = SparkSession.builder\
                    .appName('PhanMinhHung_Thesis')\
                    .master("local[*]")\
                    .config("spark.executor.memory", "12gb")\
                    .getOrCreate()
	testData = spark.read.csv(link, header = True, inferSchema = True)
	drop_list = ['_c0']
	testData = testData.select([column for column in testData.columns if column not in drop_list])
	testData = testData.withColumn('Book-Rating', testData["Book-Rating"].cast("double"))
	testData = testData.withColumn('Should-Read', testData["Should-Read"].cast("double"))
	pipelineFit = PipelineModel.load(r'\pipeline')
	yes = testData.filter(functions.col('Should-Read')== 1)
	no = testData.filter(functions.col('Should-Read')== 0)
	testData = yes.unionAll(no)
	testData.printSchema()
	testData = pipelineFit.transform(testData)
	testData.show()
	gbtm = GBTClassificationModel.load(r'\gbtm')
	print(gbtm)
	predictions = gbtm.transform(testData)
	predictions = predictions.withColumn('Should-Read', predictions["Should-Read"].cast("double"))
	predictions = predictions.withColumn('prediction', predictions["prediction"].cast("double"))
	predictions = predictions.na.drop()
	results = predictions.select('Should-Read', 'prediction')
	results.printSchema()
	y_true = np.array(results.select("Should-Read").collect())
	y_pred = np.array(results.select("prediction").collect())
	target_names = ['Should Read', 'Shouldn\'t Read']
	print(confusion_matrix(y_true, y_pred))
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	print("True Positive: ", tp)
	print("False Positive: ", fp)
	print("True Negative: ", tn)
	print("False Negative: ", fn)
	accuracy = accuracy_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred)
	recall = recall_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred)
	print(classification_report(y_true, y_pred, target_names=target_names))
	print('Accuracy : ', accuracy)
	print('Precision : ', precision)
	print('Recall : ', recall)
	print('F1 : ', f1)
	print('done testing')
	result = [accuracy, precision, recall, f1]
	spark.stop()
	return result

def predictGBT(name):
	spark = SparkSession.builder\
                    .appName('PhanMinhHung_Thesis')\
                    .master("local[*]")\
                    .config("spark.executor.memory", "12gb")\
                    .getOrCreate()
	pipelineFit = PipelineModel.load(r'\pipeline')
	gbtm = GBTClassificationModel.load(r'\gbtm')
	data = [name]
	df = pd.DataFrame(data, columns = ['Book-Title']) 
	df.to_csv(r'\test.csv')
	dataset = spark.read.csv(r'\test.csv', header = True, inferSchema = True)
	dataset = dataset.select('Book-Title')
	print(dataset)
	dataset = pipelineFit.transform(dataset)
	predictions = gbtm.transform(dataset)
	predictions.show()
	prediction = predictions.select('prediction').collect()[0]['prediction']
	if prediction == 0:
		prediction = 'Shoudn\'t Read'
	else:
		prediction = 'Should Read'
	spark.stop()
	return prediction 
