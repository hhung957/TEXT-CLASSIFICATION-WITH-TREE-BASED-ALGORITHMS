from flask import Flask, redirect, url_for, request, render_template
from rf import trainRF,testRF, predictRF
from gbt import trainGBT, testGBT, predictGBT
import os
app = Flask(__name__)

@app.route('/spark')
def spark():
   return 'hello its spark'

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/rf')
def randomforest():
	model = request.args.get('model')
	if model:
		f = open('/rftrees.txt', "w")
		f.write(model)
		f.close()
		os.system("eurekatrees --trees /rftrees.txt")
	val_res = request.args.getlist('result')
	key_res = ['accuracy ', 'precision', 'recall', 'f1']
	result = dict(zip(key_res, val_res))
	prediction = request.args.get('prediction')
	#print(result)
	return render_template('rf.html', model= model, result = result, prediction = prediction)

@app.route('/gbt')
def gradientBT():
	model = request.args.get('model')
	if model:
		f = open('/gbttrees.txt', "w")
		f.write(model)
		f.close()
		os.system("eurekatrees --trees /gbttrees.txt")
	val_res = request.args.getlist('result')
	key_res = ['accuracy ', 'precision', 'recall', 'f1']
	result = dict(zip(key_res, val_res))
	prediction = request.args.get('prediction')
	return render_template('gbt.html', model= model, result = result, prediction = prediction)

@app.route('/trainrf',methods = ['POST', 'GET'])
def trainrf():
   if request.method == 'POST':
      numTrees = int(request.form['numTrees'])
      maxDepth = int(request.form['maxDepth'])
      impurity = request.form['impurity']
      print(numTrees, maxDepth, impurity)
      print(type(numTrees), type(maxDepth), type(impurity))
      model = trainRF(numTrees, maxDepth, impurity)
      return redirect(url_for('randomforest', model = model))


@app.route('/testrf',methods = ['POST', 'GET'])
def testrf():
	link = '/Desktop/'
	name = request.form['testdata']
	link = link + name
	print(link)
	result = testRF(link)
	return redirect(url_for('randomforest', result = result))

@app.route('/predictrf',methods = ['POST', 'GET'])
def predictrf():
	name = request.form['name']
	prediction = predictRF(name)
	return redirect(url_for('randomforest', prediction = prediction))

@app.route('/traingbt',methods = ['POST', 'GET'])
def traingbt():
   if request.method == 'POST':
      maxIter = int(request.form['maxIter'])
      maxDepth = int(request.form['maxDepth'])
      stepSize = float(request.form['stepSize'])
      print(maxIter, maxDepth, stepSize)
      print(type(maxIter), type(maxDepth), type(stepSize))
      model = trainGBT(maxIter, maxDepth, stepSize)
      return redirect(url_for('gradientBT', model = model))


@app.route('/testgbt',methods = ['POST', 'GET'])
def testgbt():
	link = '/Users/Admin/Desktop/'
	name = request.form['testdata']
	link = link + name
	print(link)
	result = testGBT(link)
	return redirect(url_for('gradientBT', result = result))

@app.route('/predictgbt',methods = ['POST', 'GET'])
def predictgbt():
	name = request.form['name']
	prediction = predictGBT(name)
	return redirect(url_for('gradientBT', prediction = prediction))

if __name__ == '__main__':
	app.run()
   