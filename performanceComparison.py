import readInDatasets as rid
from ComponentClassifier import ComponentClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from multiprocessing import Process, Manager
from scipy.stats import ttest_rel
from matplotlib import pyplot as plt
from VotingMethodsClassifier import VotingMethodsClassifier
import collections as cl
import pandas as pd
import numpy as np

def runSKF(dsName, numIterations, numFolds, X, y, classifierList, queue):

	# Define type of cross validation to use
	cvIterator = []
	for i in range(numIterations):
		for train_index, test_index in StratifiedKFold(n_splits=numFolds, shuffle=True, random_state=i).split(X, y):
			cvIterator.append((train_index, test_index))
		
	# Get accuracy of classifier and store results
	errorList = {}
	for clf, clfName in classifierList:

		# If classifier is GridSearchCV or RandomizedSearchCV
		if 'searchcv' in str(type(clf)).lower():
			clf.cv = cvIterator
			clf.fit(X, y)
			clf = clf.best_estimator_

		accuracies = cross_val_score(clf, X, y, cv=cvIterator)
		error = [1.0 - accuracy for accuracy in accuracies]
		errorList[clfName] = error
	queue.put((dsName, errorList))

def runAlgorithmsOnDS(encodingType, numIterations, numFolds, classifierList):

	# Retrieve all datasets
	allDatasets = rid.getAllDS(encodingType)

	# Calculate data for table
	resultsDict = cl.defaultdict(list)
	jobs = []
	manager = Manager()
	queue = manager.Queue()
	for X, y, dsName in allDatasets:

		# Output current dataset being analized
		print('\nOn: \t' + str(dsName))

		# Perform N x K-Fold Cross Validation using multiprocessing
		p = Process(target=runSKF, args=(dsName, numIterations, numFolds, X, y, classifierList, queue,))
		jobs.append(p)
		p.start()

	# Place calculations on each dataset in data structure
	for j in jobs:
		j.join()
	while not queue.empty():
		(dsName, errorList) = queue.get()
		resultsDict['Datasets'].append(dsName)

		# Add error list for each classifier
		for clfName in errorList:
			resultsDict[clfName].append(errorList[clfName])
		
	# Cast data structure as dataframe
	resultsDf = pd.DataFrame(resultsDict)
	resultsDf = resultsDf.set_index('Datasets')
	resultsDf = resultsDf.sort_index()

	# Return data
	return resultsDf

def generateGeneralizationErrorTable(resultsDf, alpha, fileName):

	# Run T-Tests to determine statistically best classifiers
	colorMatrix = []
	for dsName in resultsDf.index:
		row = resultsDf.loc[dsName]
		means = row.apply(np.mean)
		minClf = means.idxmin()
		minErrorData = row[minClf]
		colorRow = []
		for clfName in row.index:
			data = row[clfName]

			# Perform Pairwise/Dependent/Related Two-Tailed T-Test
			#	The order of arguments for this function is: sample values (ie. testing to see if different from baseline), hypothesized values (ie. baseline)
			#	Given that the P is less than alpha:
			#		We reject the null hypothesis.  The sample values are statistically different than baseline.
			#		If the T-Statistic is positive, then sample values are statistically larger than baseline.
			#		If the T-Statistic is negative, then the sample values are statistically smaller than baseline.
			tStat, p = ttest_rel(data, minErrorData)

			# If the classifier is the min classifier, or the T-Test indicates the results are not significantly worse, set as green.  Else, set as red.
			if clfName == minClf:
				colorRow.append('blue')		# If the clf with smallest error is chosen -> blue
			elif (p < alpha) and (tStat > 0):
				colorRow.append('red')		# If the diff is statistically significant and error is greater -> red
			else:
				colorRow.append('green')	# If the diff is statistically significant and error is less OR not statistically significant at all -> green
		colorMatrix.append(colorRow)

	# Add final row with total green/blue cells in each column
	totalGreenBlueCountRow = []
	totalGreenBlueCountRow_Color = []
	colorDf = pd.DataFrame(colorMatrix)
	for columnIndex in colorDf.columns:
		column = colorDf[columnIndex]
		colorCount = cl.Counter(column)
		greenBlueCount = colorCount['green'] + colorCount['blue']
		totalGreenBlueCountRow.append(greenBlueCount)
		totalGreenBlueCountRow_Color.append('white')
	colorMatrix.append(totalGreenBlueCountRow_Color)

	# Create the actual plot of a table
	resultsDf = resultsDf.applymap(np.mean)
	vals = list(np.around(resultsDf.values, 5))
	vals.append(totalGreenBlueCountRow)
	rowLabels = resultsDf.index.tolist()
	rowLabels.append('Totals')
	fig = plt.figure(figsize=(25, 25))
	ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])
	table = ax.table(cellText=vals, cellColours=colorMatrix,
				rowLabels=rowLabels, colLabels=resultsDf.columns, 
				colWidths = [0.03]*len(vals[0]), loc='center right')
	table.set_fontsize(14)
	table.scale(2, 2)
	plt.tight_layout()
	plt.show()
	fig.savefig(fileName, bbox_inches='tight')

def generateErrorReductionRateTable(resultsDf, fileName):

	# Take the mean of each cell
	resultsDf = resultsDf.applymap(np.mean)

	# Calculate the Error Reduction Rate for each cell
	# Where the Error Reduction Rate = 1.0 - (Error of Classifier / Error of Baseline Classifier)
	resultsDf = resultsDf.apply(lambda x: 1.0 - (x / x['baseClassifier']), axis=1)

	# Add a row at the bottom of the dataframe with the average Error Reduction Rates for each algorithm
	resultsDf.loc['Average'] = resultsDf.mean()

	# Create table from dataframe
	vals = list(np.around(resultsDf.values, 5))
	fig = plt.figure(figsize=(25, 25))
	ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])
	table = ax.table(cellText=vals,
				rowLabels=resultsDf.index, colLabels=resultsDf.columns, 
				colWidths = [0.03]*len(vals[0]), loc='center right')
	table.set_fontsize(14)
	table.scale(2, 2)
	plt.tight_layout()
	plt.show()
	fig.savefig(fileName, bbox_inches='tight')

def main():

	# Initialize other variables
	numIterations = 10
	numFolds = 10
	alpha = 0.01

	# Initialize classifier hyper-parameters
	ensembleSize = 19
	criterion = 'entropy'
	attributeFilteringCoeff = 0.33
	s = 0.5
	minkowskyMinMax = (1, 3)
	kMinMax_1 = (5, 5)
	kMinMax_2 = (1, 9)
	bootstrap = True
	
	'''
	For both One Hot Encoding and VDM Encoding, do the following:

		1) Create a table comparing the error of the KNN baseline classifier to that of the FASBIR classifiers with each type of voting with both
			perturbed and non-perturbed K values.
				-All of the voting mechanisms are described in detail at the top of VotingMethodsClassifier.py
		2) For each dataset in the table created in #1, identify the best classifier and color its cell blue.
			For every other classifier, run a Pairwise T-Test to determine wether the error is statistically worse.
			If the error is statistically worse, color the cell red.  If it is not statistically worse, color the cell green.
		3) At the bottom of the table created in #1 and #2, create a row called "Totals" with the total number of green and blue
			cells in each classifier's column.
		4) Create a table comparing the error reduction rates for every error value in the table from #1.
			Error Reduction Rate of Algo = 1 - (Error of Algo / Error of KNN Baseline)
		5) At the bottom of the table created in #4, create a row called "Average" with the average of the error reduction rates in each
			classifier's column.  

		*These measurements comparing the classifiers were learned from Zhou and Yu's paper "Ensembling Local Learners Through Multimodal
			Petrurbation", which can be found here: https://www.researchgate.net/publication/3414055_Ensembling_Local_Learners_Through_Multimodal_Perturbation
	'''
	classifierList = [
		(KNeighborsClassifier(n_neighbors=kMinMax_1[0]), 'baseClassifier'),
		(VotingMethodsClassifier(ensembleSize=ensembleSize, votingType='majority_componentClf_majority', 
						attributeFilteringCoeff=attributeFilteringCoeff, criterion=criterion, bootstrap=bootstrap, s=s, 
							minkowskyMinMax=minkowskyMinMax, kMinMax=kMinMax_1, random_state=0), 'fasbir_1'),
		(VotingMethodsClassifier(ensembleSize=ensembleSize, votingType='majority_componentClf_bordaOrder', 
						attributeFilteringCoeff=attributeFilteringCoeff, criterion=criterion, bootstrap=bootstrap, s=s, 
							minkowskyMinMax=minkowskyMinMax, kMinMax=kMinMax_1, random_state=0), 'fasbir_2'),
		(VotingMethodsClassifier(ensembleSize=ensembleSize, votingType='majority_componentClf_bordaDistance', 
						attributeFilteringCoeff=attributeFilteringCoeff, criterion=criterion, bootstrap=bootstrap, s=s, 
							minkowskyMinMax=minkowskyMinMax, kMinMax=kMinMax_1, random_state=0), 'fasbir_3'),
		(VotingMethodsClassifier(ensembleSize=ensembleSize, votingType='majority_allK', 
						attributeFilteringCoeff=attributeFilteringCoeff, criterion=criterion, bootstrap=bootstrap, s=s, 
							minkowskyMinMax=minkowskyMinMax, kMinMax=kMinMax_1, random_state=0), 'fasbir_4'),
		(VotingMethodsClassifier(ensembleSize=ensembleSize, votingType='bordaCount_allK_bordaOrder', 
						attributeFilteringCoeff=attributeFilteringCoeff, criterion=criterion, bootstrap=bootstrap, s=s, 
							minkowskyMinMax=minkowskyMinMax, kMinMax=kMinMax_1, random_state=0), 'fasbir_5'),
		(VotingMethodsClassifier(ensembleSize=ensembleSize, votingType='bordaCount_allK_bordaDistance', 
						attributeFilteringCoeff=attributeFilteringCoeff, criterion=criterion, bootstrap=bootstrap, s=s, 
							minkowskyMinMax=minkowskyMinMax, kMinMax=kMinMax_1, random_state=0), 'fasbir_6'),
		(VotingMethodsClassifier(ensembleSize=ensembleSize, votingType='majority_componentClf_majority', 
						attributeFilteringCoeff=attributeFilteringCoeff, criterion=criterion, bootstrap=bootstrap, s=s, 
							minkowskyMinMax=minkowskyMinMax, kMinMax=kMinMax_2, random_state=0), 'fasbir_1_Ks'),
		(VotingMethodsClassifier(ensembleSize=ensembleSize, votingType='majority_componentClf_bordaOrder', 
						attributeFilteringCoeff=attributeFilteringCoeff, criterion=criterion, bootstrap=bootstrap, s=s, 
							minkowskyMinMax=minkowskyMinMax, kMinMax=kMinMax_2, random_state=0), 'fasbir_2_Ks'),
		(VotingMethodsClassifier(ensembleSize=ensembleSize, votingType='majority_componentClf_bordaDistance', 
						attributeFilteringCoeff=attributeFilteringCoeff, criterion=criterion, bootstrap=bootstrap, s=s, 
							minkowskyMinMax=minkowskyMinMax, kMinMax=kMinMax_2, random_state=0), 'fasbir_3_Ks'),
		(VotingMethodsClassifier(ensembleSize=ensembleSize, votingType='majority_allK', 
						attributeFilteringCoeff=attributeFilteringCoeff, criterion=criterion, bootstrap=bootstrap, s=s, 
							minkowskyMinMax=minkowskyMinMax, kMinMax=kMinMax_2, random_state=0), 'fasbir_4_Ks'),
		(VotingMethodsClassifier(ensembleSize=ensembleSize, votingType='bordaCount_allK_bordaOrder', 
						attributeFilteringCoeff=attributeFilteringCoeff, criterion=criterion, bootstrap=bootstrap, s=s, 
							minkowskyMinMax=minkowskyMinMax, kMinMax=kMinMax_2, random_state=0), 'fasbir_5_Ks'),
		(VotingMethodsClassifier(ensembleSize=ensembleSize, votingType='bordaCount_allK_bordaDistance', 
						attributeFilteringCoeff=attributeFilteringCoeff, criterion=criterion, bootstrap=bootstrap, s=s, 
							minkowskyMinMax=minkowskyMinMax, kMinMax=kMinMax_2, random_state=0), 'fasbir_6_Ks'),
	]
	
	# Generate data from one hot encoding
	resultsDf = runAlgorithmsOnDS('onehot', numIterations, numFolds, classifierList)
	
	# Generate table of classifier generalization error
	generateGeneralizationErrorTable(resultsDf, alpha, 'GeneralizationErrorTable_Onehot.png')
	
	# Generate a table of error reduction rates
	generateErrorReductionRateTable(resultsDf, 'ErrorReductionRateTable_Onehot.png')
	
	# Generate data from VDM encoding
	resultsDf = runAlgorithmsOnDS('vdm', numIterations, numFolds, classifierList)

	# Generate table of classifier generalization error
	generateGeneralizationErrorTable(resultsDf, alpha, 'GeneralizationErrorTable_VDM.png')

	# Generate a table of error reduction rates
	generateErrorReductionRateTable(resultsDf, 'ErrorReductionRateTable_VDM.png')
	

	'''
	The previously created tables demonstrate that the ensemble algorithm can give results that are statistically superior
		to the KNN baseline classifier.  One might compare the errors of the baseline KNN and the ensemble methods, and 
		feel that although the improvements are statistically significant, they are miniscule (ie. 2 to 3% error reduction).
		To demonstrate the superiority of the ensemble methods, I have created a second table of errors and error reduction,
		this time comparing the KNN baseline, the classifier from the original One Hot Encoded tables that performed best,
		and the classifier from the original One Hot Encoded tables that performed best... with some of the hyper-parameters
		tuned using RandomSearchCV.

	The purpose of this second table is to serve as a quick and obvious demonstration that the ensemble methods offer massive
		potential improvements over the KNN baseline.  By simply roughly tuning a few of the hyperparameters, we achieve a
		massive improvement in error reduction from ~9.9% to ~23.1%.  This while not even toughing the most important hyper-
		parameter of all:  ensemble size.  Additionally, increasing the n_iter parameter for RandomizedSearchCV would obviously
		lead to further improvements as well.

	Simply, the purpose of this table is to show that if crude hyper-parameter tuning can yield huge improvements, then a more 
		calibrated and patient tuning will yield improvements at least as huge, and probably even greater.
	'''
	# Demonstrate potential of VotingMethodsClassifier with tuned hyper-parameters
	classifierList = [
		(KNeighborsClassifier(n_neighbors=kMinMax_1[0]), 'baseClassifier'),
		(VotingMethodsClassifier(ensembleSize=ensembleSize, votingType='bordaCount_allK_bordaOrder', criterion=criterion, 
						attributeFilteringCoeff=attributeFilteringCoeff, bootstrap=bootstrap, s=s, 
							minkowskyMinMax=minkowskyMinMax, kMinMax=kMinMax_1, random_state=0), 'fasbir_5'),
		(RandomizedSearchCV(
			estimator=VotingMethodsClassifier(votingType='bordaCount_allK_bordaOrder', criterion=criterion, bootstrap=bootstrap, random_state=0), 
			param_distributions={
				'ensembleSize': [19],
				'attributeFilteringCoeff': [float(i) for i in np.linspace(0.2, 1.0, num=1000)],
				's': [float(i) for i in np.linspace(0.3, 0.7, num=1000)],
				'minkowskyMinMax': [(1, 3), (1, 4), (1, 5)],
				'kMinMax': [(1, 1), (3, 3), (5, 5), (7, 7), (9, 9), (11, 11), (13, 13), (15, 15)]
			}, 
			n_iter=10, n_jobs=-1, random_state=0), 'fasbir_5_rscv')
	]

	# Generate a table for one hot encoding
	resultsDf = runAlgorithmsOnDS('onehot', numIterations, numFolds, classifierList)
	
	# Generate table of classifier generalization error
	generateGeneralizationErrorTable(resultsDf, alpha, 'GeneralizationErrorTable_RandomizedSearchCV_Onehot.png')

	# Generate a table of error reduction rates
	generateErrorReductionRateTable(resultsDf, 'ErrorReductionRateTable_RandomizedSearchCV_Onehot.png')
	

if __name__ == '__main__':
	main()

