'''

================================================================================================================================

Six voting methods have been implemented here.  Generally speaking, the naming convention for the voting
	methods is as follows:

	*Ensemble Voting Technique*_*Ensemble Data Source*_*Metric Info For Ensemble Data Source*

I'll give two examples:

	If the voting method is:  "majority_componentClf_majority"

		Then the ensemble determines its prediction by majority voting.
		The source of the data on which the ensemble does majority voting is the component classifiers.
		The component classifiers make their individual predictions by majority voting of the k nearest points.

	If the voting method is:  "bordaCount_allK_bordaDistance"

		Then the ensemble determines its prediction by borda counting.
		The source of the data on which the ensemble performs borda counting is the set of
			the k nearest neighbors for each component classifier, ie. "allK".
		The count of each point is determined by one divided by its distance metric.

================================================================================================================================

A brief explanation of each of the voting methods is as follows:

1)	majority_componentClf_majority
		Predictions are made by a simple majority vote of the predictions of the component classifiers.

2)	majority_componentClf_bordaOrder
		Predictions are made by a simple majority vote of the predictions of the component classifiers,
		where the predictions of the component classifiers are determined by borda counting with regard 
		to relative nearest neighbor order.

		The borda counts of each component classifier are determined as follows:
			If k = 3, and the first and third nearest neighbors are of class A, and the second is of class B, then:
				Class A Count = 3 + 1
				Class B Count = 2
			...Therefore, Class A is the predicted.
			Where class count is the sum of (k - (# of nearest neighbor - 1)).
			So if point X is the 4th nearest neighbor, and k = 9, then X Count = (9 - (4 - 1)) = 6.

3)	majority_componentClf_bordaDistance
		Predictions are made by a simple majority vote of the predictions of the component classifiers,
		where the predictions of the component classifiers are determined by borda counting.  In this 
		case, count is not based on relative order as the I-th nearest point, but on one divided by the
		distance measure for each point.

		The borda counts of each component classifier are determined as follows:
			If k = 3. and the first and third nearest neighbors are of class A, and the secone is of class B, then:
				Class A Count = [1 / (distance of first nearest neighbor)] + [1 / (distance of third nearest neighbor)]
				Class B Count = [1 / (distance of second nearest neighbor)]
			...The class with the larger count is predicted.

4)	majority_allK
		Predictions are made by a simple majority vote of the set composed of the k nearest neighbors 
		of each component classifier.

		If the component classifiers' k nearest neighbors were:
			Component Clf #1: 	{Class A, Class A, Class A}
			Component Clf #2:	{Class B, Class A, Class B}
			Component Clf #3:	{Class A, Class B, Class B}
		...Then the total votes for the ensemble would be {Class A: 5, Class B: 4}, and class A is predicted.
	
5)	bordaCount_allK_bordaOrder
		Predictions are made by a borda count of the results returned by component classifiers, where
		the results of component classifiers are determined by borda counting with regard to relative
		nearest neighbor order.

		The borda counts of each component classifier are determined as follows:
			If k = 3, and the first and third nearest neighbors are of class A, and the second is of class B, then:
				Class A Count = 3 + 1
				Class B Count = 2
			...Therefore, {Class A: 4, Class B: 2} is returned, and a borda count is performed over
				all results of component classifiers.
			Where class count is the sum of (k - (# of nearest neighbor - 1)).
			So if point X is the 4th nearest neighbor, and k = 9, then X Count = (9 - (4 - 1)) = 6.

		If the component classifier results for the ensemble were:
			Component Clf #1:	{Class A: 4, Class B: 2}	
			Component Clf #2:	{Class A: 2, Class B: 4}
			Component Clf #3:	{Class A: 6, Class B: 0}
		...Then the borda vote of the ensemble would be {Class A: 12, Class B: 6}, and Class A is predicted.

6)	bordaCount_allK_bordaDistance
		Predictions are made by a borda count of the results returned by component classifiers, where
		the results of component classifiers are determined by borda counting.  In this case, count 
		is not based on relative order as the I-th nearest point, but on one divided by the distance 
		measure for each point.

		The borda counts of each component classifier are determined as follows:
			If k = 3. and the first and third nearest neighbors are of class A, and the secone is of class B, then:
				Class A Count = [1 / (distance of first nearest neighbor)] + [1 / (distance of third nearest neighbor)]
				Class B Count = [1 / (distance of second nearest neighbor)]
			...Therefore, {Class A: *some count*, Class B: *some count*} is returned, and a borda count 
				is performed over all results of component classifiers.

		If the component classifier results for the ensemble were:
			Component Clf #1:	{Class A: 4, Class B: 2}	
			Component Clf #2:	{Class A: 2, Class B: 4}
			Component Clf #3:	{Class A: 6, Class B: 0}
		...Then the borda vote of the ensemble would be {Class A: 12, Class B: 6}, and Class A is predicted.

'''

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
from ComponentClassifier import ComponentClassifier
import collections as cl
import pandas as pd
import numpy as np

class VotingMethodsClassifier(BaseEstimator, ClassifierMixin):

	def __init__(self, ensembleSize=19, votingType='majority_componentClf_majority', criterion='entropy', attributeFilteringCoeff=None, bootstrap=True, s=1.0, 
			minkowskyMinMax=(2, 2), kMinMax=(5, 5), random_state=None):
		self.ensembleSize = ensembleSize
		self.votingType = votingType
		self.criterion = criterion
		self.attributeFilteringCoeff = attributeFilteringCoeff
		self.bootstrap = bootstrap
		self.s = s
		self.minkowskyMinMax = minkowskyMinMax
		self.kMinMax = kMinMax
		self.random_state = random_state

	def fit(self, X, y):

		# Check that parameters are formatted properly
		X, y = check_X_y(X, y)
		assert (type(self.ensembleSize) == int), "ensembleSize parameter must be int"
		assert ((type(self.minkowskyMinMax) == tuple) and (len(self.minkowskyMinMax) == 2)), "minkowskyMinMax must be a tuple of length 2"
		assert ((type(self.kMinMax) == tuple) and (len(self.minkowskyMinMax) == 2)), "kMinMax must be a tuple of length 2"
		assert (1 <= self.ensembleSize), "ensembleSize parameter must be greater than or equal to 1"

		# Generate component classifiers
		componentClassifiers = []
		for i in range(self.ensembleSize):
			if self.random_state is None:
				random_state = self.random_state
			else:
				random_state = self.random_state + i
			subClf = ComponentClassifier(criterion=self.criterion, attributeFilteringCoeff=self.attributeFilteringCoeff, bootstrap=self.bootstrap, s=self.s, 
							minkowskyMin=min(self.minkowskyMinMax), minkowskyMax=max(self.minkowskyMinMax), 
								kMin=min(self.kMinMax), kMax=max(self.kMinMax), votingType=self.votingType, random_state=random_state)
			subClf.fit(X, y)
			componentClassifiers.append(subClf)
		self.componentClassifiers_ = componentClassifiers

		# Create class variable of classification class values
		self.classes_ = unique_labels(y)

		return self

	def predict(self, X):

		# Check that classifier has been fitted
		try:
			getattr(self, "componentClassifiers_")
		except AttributeError:
			raise NotFittedError("You must fit classifer before using predict!")

		# Convert X to numpy array
		X = check_array(X)

		# Generate predictions based on voting method
		componentPredictions = []
		for subClf in self.componentClassifiers_:
			predictions = subClf.predict(X)
			componentPredictions.append(predictions)
		df = pd.DataFrame(data=componentPredictions)
		if self.votingType == 'majority_componentClf_majority':
			votingPredictions = []
			for index in df.columns:
				column = df[index]
				choice = cl.Counter(column).most_common(1)[0][0]
				votingPredictions.append(choice)
			predictions = np.asarray(votingPredictions)
			return predictions
			
		elif self.votingType == 'majority_componentClf_bordaOrder':
			votingPredictions = []
			for index in df.columns:
				column = df[index]
				choice = cl.Counter(column).most_common(1)[0][0]
				votingPredictions.append(choice)
			predictions = np.asarray(votingPredictions)
			return predictions

		elif self.votingType == 'majority_componentClf_bordaDistance':
			votingPredictions = []
			for index in df.columns:
				column = df[index]
				choice = cl.Counter(column).most_common(1)[0][0]
				votingPredictions.append(choice)
			predictions = np.asarray(votingPredictions)
			return predictions

		elif self.votingType == 'majority_allK':
			votingPredictions = []
			for index in df.columns:
				column = df[index]
				column = column.tolist()
				column = [element for sublist in column for element in sublist]
				choice = cl.Counter(column).most_common(1)[0][0]
				votingPredictions.append(choice)
			predictions = np.asarray(votingPredictions)
			return predictions
		
		elif self.votingType == 'bordaCount_allK_bordaOrder':
			votingPredictions = []
			for index in df.columns:
				column = df[index]
				column = column.tolist()
				column = np.asarray(column)
				column = column.flatten()
				total = cl.Counter()
				for subCounter in column:
					total = total + subCounter
				choice = total.most_common(1)[0][0]
				votingPredictions.append(choice)
			predictions = np.asarray(votingPredictions)
			return predictions

		elif self.votingType == 'bordaCount_allK_bordaDistance':
			votingPredictions = []
			for index in df.columns:
				column = df[index]
				column = column.tolist()
				column = np.asarray(column)
				column = column.flatten()
				total = cl.Counter()
				for subCounter in column:
					total = total + subCounter
				choice = total.most_common(1)[0][0]
				votingPredictions.append(choice)
			predictions = np.asarray(votingPredictions)
			return predictions			

	def score(self, X, y, sample_weight=None):

		# Check that classifier has been fitted
		try:
			getattr(self, "componentClassifiers_")
		except AttributeError:
			raise NotFittedError("You must fit classifer before using predict!")

		# Convert X to numpy array
		X = check_array(X)

		# Generate predictions
		predictions = self.predict(X)

		# Calculate score
		score = accuracy_score(y, predictions)

		return score

def main():

	'''
	The following code demonstrates how to use the VotingMethodsClassifier class.
	'''

	# First, create some dummy data
	import random

	numDataPoints = 1000
	numFeatures = 5

	X_train = [[j for j in range(numFeatures)] for i in range(numDataPoints)]
	y_train = [random.randint(0, 1) for i in range(numDataPoints)]

	X_test = [[j for j in range(numFeatures)] for i in range(numDataPoints)]
	y_test = [random.randint(0, 1) for i in range(numDataPoints)]

	# Next, apply the classifier
	clf = VotingMethodsClassifier(ensembleSize=19, attributeFilteringCoeff=0.33, criterion='entropy', bootstrap=True, s=0.5, 
					minkowskyMinMax=(1, 3), kMinMax=(11, 11), random_state=None)
	clf.fit(X_train, y_train)
	predictions = clf.predict(X_test)
	score = clf.score(X_test, y_test)

	print('')
	print(predictions)
	print('')
	print(score)

	'''
	The following code checks whether the SubsetKNN estimator adheres to scikit-learn conventions.
	'''

	from sklearn.utils.estimator_checks import check_estimator
	check_estimator(VotingMethodsClassifier)  # passes


if __name__ == '__main__':
	main()

