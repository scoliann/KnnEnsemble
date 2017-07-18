from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
import collections as cl
import pandas as pd
import numpy as np
import random

class ComponentClassifier(BaseEstimator, ClassifierMixin):

	def __init__(self, criterion='entropy', attributeFilteringCoeff=None, bootstrap=True, s=1.0, minkowskyMin=2, minkowskyMax=2, 
			kMin=5, kMax=5, votingType='majority_componentClf_majority', random_state=None):
		self.criterion = criterion
		self.attributeFilteringCoeff = attributeFilteringCoeff
		self.bootstrap = bootstrap
		self.s = s
		self.minkowskyMin = minkowskyMin
		self.minkowskyMax = minkowskyMax
		self.kMin = kMin
		self.kMax = kMax
		self.votingType = votingType
		self.random_state = random_state

	def _cleanFeatures(self, X):

		# Clean data to include only features from feature subset
		droppedColumns = [index for index in range(len(self.useFeatures_)) if self.useFeatures_[index] == 0]
		X = pd.DataFrame(X)
		X = X.drop(droppedColumns, axis=1)

		return X

	def fit(self, X, y):

		# Check that parameters are formatted properly
		X, y = check_X_y(X, y)
		assert ((self.attributeFilteringCoeff == None) or type(self.attributeFilteringCoeff) == float), "attributeFilteringCoeff parameter must be None or float"
		assert (type(self.bootstrap) == bool), "bootstrap parameter must be bool"
		assert ((self.s == None) or type(self.s) == float), "s parameter must be None or float"
		assert (type(self.minkowskyMin) == int), "minkowskyMin parameter must be int"
		assert (type(self.minkowskyMax) == int), "minkowskyMax parameter must be int"
		assert (type(self.kMin) == int), "kMin parameter must be int"
		assert (type(self.kMax) == int), "kMax parameter must be int"
		assert (type(self.votingType) == str), "votingType parameter must be str"
		assert ((self.random_state == None) or (type(self.random_state) == int)), "random_state parameter must be None or int"
		assert (self.criterion in ['entropy', 'gini']), "criterion parameter must be \'entropy\' or \'gini\'"		
		assert ((self.attributeFilteringCoeff == None) or (self.attributeFilteringCoeff > 0)), "attributeFilteringCoeff parameter must be greater than or equal to 0"
		assert ((self.s == None) or ((0 < self.s) and (self.s <= 1))), "s parameter must be greater than 0 and less than or equal to 1"
		assert (0 < self.minkowskyMin), "minkowskyMin parameter must be greater than 0"
		assert (0 < self.minkowskyMax), "minkowskyMax parameter must be greater than 0"
		assert (self.minkowskyMin <= self.minkowskyMax), "minkowskyMin parameter must be less than or equal to minkowskyMax parameter"
		assert (1 <= self.kMin), "kMin parameter must be greater than or equal to 1"
		assert (1 <= self.kMax), "kMax parameter must be greater than or equal to 1"
		assert (self.kMin <= self.kMax), "kMin parameter must be less than or equal to kMax parameter"
		assert (self.votingType in ['majority_componentClf_majority', 'majority_componentClf_bordaOrder', 'majority_componentClf_bordaDistance', 'majority_allK', 
				'bordaCount_allK_bordaOrder', 'bordaCount_allK_bordaDistance']), "votingType parameter must be of a recognized type"

		# Seed random number generator
		random.seed(self.random_state)

		# Apply attribute filtering
		if self.attributeFilteringCoeff != None:

			# Get information gain
			igClf = DecisionTreeClassifier(criterion=self.criterion, random_state=self.random_state)
			igClf.fit(X, y)
			featureInformationGains = igClf.feature_importances_

			# Filter which features according to information gain
			avgInformationGains = np.mean(featureInformationGains)
			f = self.attributeFilteringCoeff * avgInformationGains
			validFeatures = [0 if gain < f else 1 for gain in featureInformationGains]

			# If no features made the cutoff, set a random feature as valid
			if 1 not in validFeatures:
				randomIndex = random.choice([i for i in range(len(validFeatures))])
				validFeatures[randomIndex] = 1
		else:
			validFeatures = [1 for i in range(len(X[0]))]

		# Apply feature selection
		validIndicies = [index for index in range(len(validFeatures)) if validFeatures[index] == 1]
		floatNumChoose = self.s * len(validIndicies)
		decimalNumChoose = str(floatNumChoose).split('.')[1]
		potentialNumChoose = random.choice([int(floatNumChoose), int(floatNumChoose + 1)]) if decimalNumChoose == '5' else round(floatNumChoose)
		numChoose = 1 if potentialNumChoose == 0 else potentialNumChoose
		chosenIndicies = random.sample(validIndicies, numChoose)
		useFeatures = [1 if index in chosenIndicies else 0 for index in range(len(validFeatures))]
		self.useFeatures_ = useFeatures

		# Get a bootstrapped sample of the data
		if self.bootstrap == True:
			X, y = resample(X, y, random_state=self.random_state)

		# Generate an odd K for even number class problem and viceversa
		if self.kMin == self.kMax:
			self.n_neighbors_ = self.kMin
		else:
			if len(unique_labels(y)) % 2 == 0:
				self.n_neighbors_ = random.choice([k for k in range(self.kMin, self.kMax + 1) if k % 2 == 1])
			else:
				self.n_neighbors_ = random.choice([k for k in range(self.kMin, self.kMax + 1) if k % 2 == 0])

		# Generate a value for the order of the Minkowsky distance
		minkowskyOrder = random.randint(self.minkowskyMin, self.minkowskyMax)

		# Clean data to include only features from feature subset
		X = self._cleanFeatures(X)

		# Initialize and train KNN classifier on feature subset
		knn = KNeighborsClassifier(n_neighbors=self.n_neighbors_, p=minkowskyOrder)
		knn.fit(X, y)
		self.y_ = y
		self.knn_ = knn

		# Create class variable of classification class values
		self.classes_ = unique_labels(y)

		return self

	def predict(self, X):

		# Check that classifier has been fitted
		try:
			getattr(self, "knn_")
		except AttributeError:
			raise NotFittedError("You must fit classifer before using predict!")

		# Convert X to numpy array
		X = check_array(X)

		# Clean data to include only features from feature subset
		X = self._cleanFeatures(X)

		# Generate predictions based on voting method
		if self.votingType == 'majority_componentClf_majority':
			predictions = self.knn_.predict(X)
			return predictions

		elif self.votingType == 'majority_componentClf_bordaOrder':
			(distances, indicies) = self.knn_.kneighbors(X, self.n_neighbors_)
			predictions = []
			for indexList in indicies:
				counter = cl.Counter()
				for i in range(len(indexList)):
					index = indexList[i]
					iclass = self.y_[index]
					points = len(indexList) - i
					counter[iclass] += points
				choice = counter.most_common(1)[0][0]
				predictions.append(choice)
			return predictions

		elif self.votingType == 'majority_componentClf_bordaDistance':
			(distances, indicies) = self.knn_.kneighbors(X, self.n_neighbors_)
			predictions = []
			for i in range(len(indicies)):
				indexList = indicies[i]
				distanceList = distances[i]
				counter = cl.Counter()
				for j in range(len(indexList)):
					index = indexList[j]
					iclass = self.y_[index]
					distance = distanceList[j]
					counter[iclass] += (1.0 /  (distance + 1.0))
				choice = counter.most_common(1)[0][0]
				predictions.append(choice)
			return predictions

		elif self.votingType == 'majority_allK':
			(distances, indicies) = self.knn_.kneighbors(X, self.n_neighbors_)
			predictions = [[self.y_[index] for index in indexList] for indexList in indicies]
			return predictions

		elif self.votingType == 'bordaCount_allK_bordaOrder':
			(distances, indicies) = self.knn_.kneighbors(X, self.n_neighbors_)
			predictions = []
			for indexList in indicies:
				counter = cl.Counter()
				for i in range(len(indexList)):
					index = indexList[i]
					iclass = self.y_[index]
					points = len(indexList) - i
					counter[iclass] += points
				predictions.append(counter)
			return predictions

		elif self.votingType == 'bordaCount_allK_bordaDistance':
			(distances, indicies) = self.knn_.kneighbors(X, self.n_neighbors_)
			predictions = []

			for i in range(len(indicies)):
				indexList = indicies[i]
				distanceList = distances[i]
				counter = cl.Counter()
				for j in range(len(indexList)):
					index = indexList[j]
					iclass = self.y_[index]
					distance = distanceList[j]
					counter[iclass] += (1.0 / (distance + 1.0))
				predictions.append(counter)
			return predictions

	def score(self, X, y, sample_weight=None):

		# Check that classifier has been fitted
		try:
			getattr(self, "knn_")
		except AttributeError:
			raise NotFittedError("You must fit classifer before using predict!")

		# Convert X to numpy array
		X = check_array(X)

		# Clean data to include only features from feature subset
		X = self._cleanFeatures(X)

		return self.knn_.score(X, y, sample_weight=sample_weight)

def main():

	'''
	The following code demonstrates how to use the ComponentClassifier class.
	'''

	# First, create some dummy data
	numDataPoints = 1000
	numFeatures = 5

	X_train = [[j for j in range(numFeatures)] for i in range(numDataPoints)]
	y_train = [random.randint(0, 1) for i in range(numDataPoints)]

	X_test = [[j for j in range(numFeatures)] for i in range(numDataPoints)]
	y_test = [random.randint(0, 1) for i in range(numDataPoints)]

	# Next, apply the classifier
	clf = ComponentClassifier(criterion='entropy', attributeFilteringCoeff=0.33, bootstrap=True, s=0.5, minkowskyMin=1, minkowskyMax=3, 
					kMin=11, kMax=11, votingType='majority_componentClf_majority', random_state=None)
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
	check_estimator(ComponentClassifier)  # passes


if __name__ == '__main__':
	main()

