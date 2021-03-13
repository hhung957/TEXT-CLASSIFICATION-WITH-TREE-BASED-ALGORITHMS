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
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import RandomForestClassificationModel
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support,accuracy_score,f1_score,recall_score,precision_score

def trainRF(numTrees, maxDepth,impurity):
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
	#Random Forest 
	rf = RandomForestClassifier(labelCol="Should-Read", featuresCol="features")
	rf.setNumTrees(numTrees)
	rf.setMaxDepth(maxDepth)
	rf.setImpurity(impurity)
	print(rf)
	rfModel = rf.fit(trainingData)
	rfModel.write().overwrite().save('/rfm')
	print(rfModel)
	model = rfModel.toDebugString
	print('done training')
	spark.stop()
	return model
def testRF(link):
	spark = SparkSession.builder\
                    .appName('PhanMinhHung_Thesis')\
                    .master("local[*]")\
                    .config("spark.executor.memory", "12gb")\
                    .getOrCreate()
	testData = spark.read.csv(link, header = True, inferSchema = True)
	drop_list = ['_c0']
	testData = testData.select([column for column in testData.columns if column not in drop_list])
	testData = testData.withColumn('Should-Read', testData["Should-Read"].cast("double"))
	yes = testData.filter(functions.col('Should-Read')== 1)
	no = testData.filter(functions.col('Should-Read')== 0)
	testData = yes.unionAll(no)
	pipelineFit = PipelineModel.load(r'\pipeline')
	testData.printSchema()
	testData = pipelineFit.transform(testData)
	testData.show()
	rfm = RandomForestClassificationModel.load(r'\rfm')
	print(rfm)
	predictions = rfm.transform(testData)
	predictions = predictions.withColumn('Should-Read', predictions["Should-Read"].cast("double"))
	predictions = predictions.withColumn('prediction', predictions["prediction"].cast("double"))
	predictions = predictions.na.drop()
	results = predictions.select('Should-Read', 'prediction')
	results.printSchema()
	y_true = np.array(results.select("Should-Read").collect())
	y_pred = np.array(results.select("prediction").collect())
	target_names = ['Should Read', 'Shouldn\'t Read']
	#target_names = ['rating 0', 'rating 1', 'rating 2', 'rating 3']
	#print(classification_report(y_true, y_pred, target_names=target_names))
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

def predictRF(name):
	spark = SparkSession.builder\
                    .appName('PhanMinhHung_Thesis')\
                    .master("local[*]")\
                    .config("spark.executor.memory", "12gb")\
                    .getOrCreate()
	pipelineFit = PipelineModel.load(r'\pipeline')
	rfm = RandomForestClassificationModel.load(r'\rfm')	
	data = [name]
	df = pd.DataFrame(data, columns = ['Book-Title']) 
	df.to_csv(r'\test.csv')
	dataset = spark.read.csv(r'\test.csv', header = True, inferSchema = True)
	dataset = dataset.select('Book-Title')
	print(dataset)
	dataset = pipelineFit.transform(dataset)
	predictions = rfm.transform(dataset)
	predictions.show()
	prediction = predictions.select('prediction').collect()[0]['prediction']
	if prediction == 0:
		prediction = 'Shoudn\'t Read'
	else:
		prediction = 'Should Read'
	spark.stop()
	return prediction 
