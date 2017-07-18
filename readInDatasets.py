'''
About this file

Datasets are stored in different formats by different researchers.
The purpose of this file is to provide a function to read in each dataset, do some minor processing, and put the data in a common format.
The continuous features in every dataset are normalized between 0 and 1 inclusive.  
This is done so that the weighting of each feature (via the distance measure) on classification is equal.
Discreate features can be encoded using One Hot Encoding or VDM Encoding.
	-VDM Encoding encodes discrete features such that normal application of Minkowsky distance on the feature set will calculate 
		the Minkovdm distance metric described by Zhou and Yu in "Ensembling Local Learners Through Multimodal Perturbation".
This process will make it easiest to import the datasets and test the performance of the KNN ensemble classifier.
'''

import pandas as pd
import collections as cl
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y

'''
Reads in dataset using one hot encoding on categorical features
'''
def oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies, removeIndicies=[], delimiter=','):

	# Read in data from CSV
	# Remove columns to ignore
	# Set all non categorical columns as numerical
	dfData = pd.read_csv(datasetPath, header=None, sep=delimiter)
	dfData = dfData.drop(removeIndicies, axis=1)
	for columnIndex in dfData.columns:
		if columnIndex not in (categoricalFeaturesIndicies + [classColumnIndex]):
			dfData[columnIndex] = pd.to_numeric(dfData[columnIndex], errors='coerce')
	dfData = dfData.dropna()

	# Encode the classification labels to numbers
	# Get classes and one hot encoded feature vectors
	le = LabelEncoder()
	y = le.fit_transform(dfData[classColumnIndex])
	X = dfData.drop([classColumnIndex], axis=1)
	X = pd.get_dummies(X)

	# Normalize dataset so all features are [0.0, 1.0]
	X = (X - X.min()) / (X.max() - X.min())
	X = X.dropna(axis=1, how='all')

	# Check the format of the data
	X, y = check_X_y(X, y)

	# Return X and y
	return X, y

'''
Reads in dataset using VDM proportions in place of categorical features
'''
def VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies, removeIndicies=[], delimiter=','):

	# Read in data from CSV
	# Remove columns to ignore
	# Set all non categorical columns as numerical
	dfData = pd.read_csv(datasetPath, header=None, sep=delimiter)
	dfData = dfData.drop(removeIndicies, axis=1)
	for columnIndex in dfData.columns:
		if columnIndex not in (categoricalFeaturesIndicies + [classColumnIndex]):
			dfData[columnIndex] = pd.to_numeric(dfData[columnIndex], errors='coerce')
	dfData = dfData.dropna()

	# Apply Pseudo-VDM Encoding to Categorical Variables
	VDM_Features = cl.defaultdict(list)
	for columnIndex in categoricalFeaturesIndicies:

		# Calculate Nau
		Nau_Dict = cl.Counter(dfData[columnIndex])

		# For each Nau calculate Nauc values
		Nauc_Dict = {}
		for categoricalFeature in Nau_Dict.keys():
			classes = dfData.loc[dfData[columnIndex] == categoricalFeature, classColumnIndex]
			Nauc_Dict[categoricalFeature] = cl.Counter(classes)

		# Iterate over the rows in the dataframe
		for rowIndex in dfData[columnIndex].index.values:
			categoricalFeature = dfData[columnIndex][rowIndex]
			Nau = float(Nau_Dict[categoricalFeature])

			# Calculate Nauc / Nau for each class as a feature
			VDM_List = []
			classList = set(dfData[classColumnIndex])
			for iclass in classList:
				Nauc = float(Nauc_Dict[categoricalFeature][iclass])
				VDM_Features[rowIndex].append(Nauc / Nau)

	# Create a dataframe from the VDM features
	vdmFeatureDf = pd.DataFrame.from_dict(VDM_Features, orient='index')

	# Drop all columns containing categorical variables
	dfData = dfData.drop(categoricalFeaturesIndicies, axis=1)

	# Encode the classification labels to numbers, and drop class column
	le = LabelEncoder()
	y = le.fit_transform(dfData[classColumnIndex])
	X = dfData.drop([classColumnIndex], axis=1)

	# Normalize dataset so all continuous features are [0.0, 1.0]
	X = (X - X.min()) / (X.max() - X.min())
	X = X.dropna(axis=1, how='all')

	# Add VDM features to X
	X = pd.concat([X, vdmFeatureDf], axis=1, ignore_index=True)

	# Check the format of the data
	X, y = check_X_y(X, y)

	# Return X and y
	return X, y

'''
Annealing Dataset
URL: https://archive.ics.uci.edu/ml/datasets/Annealing
'''
def importAnnealDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/anneal/anneal (training and test).data'
	classColumnIndex = 38
	categoricalFeaturesIndicies = [0, 1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 35, 36, 37]
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	return X, y

'''
Auto Dataset
URL: https://archive.ics.uci.edu/ml/datasets/automobile
'''
def importAutoDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/auto/imports-85.data'
	classColumnIndex = 0
	categoricalFeaturesIndicies = [2, 3, 4, 5, 6, 7, 8, 14, 15, 17]
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	return X, y

'''
Balance Scale Dataset
URL: http://archive.ics.uci.edu/ml/datasets/balance+scale
'''
def importBalanceScaleDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/balanceScale/balance-scale.data'
	classColumnIndex = 0
	categoricalFeaturesIndicies = []
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	return X, y

'''
Breast Tissue Dataset
URL: http://archive.ics.uci.edu/ml/datasets/breast+tissue
Note: data.csv is the relevant part of the data set found in the file BreastTissue.xls.
'''
def importBreastTissueDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/breastTissue/data.csv'
	classColumnIndex = 0
	categoricalFeaturesIndicies = []
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	return X, y

'''
Credit Approval Dataset
URL: http://archive.ics.uci.edu/ml/datasets/credit+approval
'''
def importCreditApprovalDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/creditApproval/crx.data'
	classColumnIndex = 15
	categoricalFeaturesIndicies = [0, 3, 4, 5, 6, 8, 9, 11, 12]
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	return X, y

'''
Diabetes Dataset
URL: https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes
'''
def importDiabetesDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/diabetes/pima-indians-diabetes.data'
	classColumnIndex = 8
	categoricalFeaturesIndicies = []
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	return X, y

'''
German Credit Dataset
URL: https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)
'''
def importGermanCreditDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/germanCredit/german.data'
	classColumnIndex = 20
	categoricalFeaturesIndicies = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
	delimiter = '\s+'	#' '
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies, delimiter=delimiter)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies, delimiter=delimiter)
	return X, y

'''
Glass Dataset
URL: https://archive.ics.uci.edu/ml/datasets/glass+identification
'''
def importGlassDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/glass/glass.data'
	classColumnIndex = 10
	categoricalFeaturesIndicies = []
	removeIndicies = [0]
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies, removeIndicies=removeIndicies)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies, removeIndicies=removeIndicies)
	return X, y

'''
Heart Dataset
URL: http://archive.ics.uci.edu/ml/datasets/statlog+(heart)
'''
def importHeartDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/heart/heart.dat'
	classColumnIndex = 13
	categoricalFeaturesIndicies = [1, 2, 5, 6, 8, 12]
	delimiter = '\s+'	#' '
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies, delimiter=delimiter)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies, delimiter=delimiter)
	return X, y

'''
Hepatitis Dataset
URL: https://archive.ics.uci.edu/ml/datasets/hepatitis
'''
def importHepatitisDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/hepatitis/hepatitis.data'
	classColumnIndex = 0
	categoricalFeaturesIndicies = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	return X, y

'''
Image Segmentation Dataset
URL: http://archive.ics.uci.edu/ml/datasets/image+segmentation
Note: segmentation.dataAndTest is a concatination of the contents of segmentation.data and segmentation.test.
'''
def importImageSegmentationDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/imageSegmentation/segmentation.dataAndTest'
	classColumnIndex = 0
	categoricalFeaturesIndicies = []
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	return X, y

'''
Indian Liver Dataset
URL: https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)
'''
def importIndianLiverDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/indianLiver/Indian Liver Patient Dataset (ILPD).csv'
	classColumnIndex = 10
	categoricalFeaturesIndicies = [1]
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	return X, y

'''
Ionosphere Dataset
URL: https://archive.ics.uci.edu/ml/datasets/ionosphere
'''
def importIonosphereDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/ionosphere/ionosphere.data'
	classColumnIndex = 34
	categoricalFeaturesIndicies = []
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	return X, y

'''
Iris Dataset
URL: https://archive.ics.uci.edu/ml/datasets/iris
'''
def importIrisDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/iris/iris.data'
	classColumnIndex = 4
	categoricalFeaturesIndicies = []
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	return X, y

'''
Liver Disorders Dataset
URL: https://archive.ics.uci.edu/ml/datasets/liver+disorders
'''
def importLiverDisordersDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/liverDisorders/bupa.data'
	classColumnIndex = 6
	categoricalFeaturesIndicies = []
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	return X, y

'''
Sonar Dataset
URL: http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)
'''
def importSonarDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/sonar/sonar.all-data'
	classColumnIndex = 60
	categoricalFeaturesIndicies = []
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	return X, y

'''
Soybean Dataset
URL: https://archive.ics.uci.edu/ml/datasets/Soybean+(Large)
Note: soybean-large.dataAndTest is a concatination of the contents of soybean-large.data and soybean-large.test.
'''
def importSoybeanDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/soybean/soybean-large.dataAndTest'
	classColumnIndex = 0
	categoricalFeaturesIndicies = [i+1 for i in range(35)]
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	return X, y

'''
Vehicle Dataset
URL: https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes)
Note: compiledVehicleDataset.data is a concatination of the contents of the xaa.dat through xai.dat files in the original dataset.
'''
def importVehicleDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/vehicle/compiledVehicleDataset.data'
	classColumnIndex = 18
	categoricalFeaturesIndicies = []
	delimiter = '\s+'	#' '
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies, delimiter=delimiter)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies, delimiter=delimiter)
	return X, y

'''
Vote Dataset
URL: https://archive.ics.uci.edu/ml/datasets/congressional+voting+records
'''
def importVoteDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/vote/house-votes-84.data'
	classColumnIndex = 0
	categoricalFeaturesIndicies = [i+1 for i in range(16)]
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies)
	return X, y

'''
Vowel Dataset
URL: https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Vowel+Recognition+-+Deterding+Data)
'''
def importVowelDS(categoricalFeatureStrategy):
	datasetPath = 'datasets/vowel/vowel-context.data'
	classColumnIndex = 13
	categoricalFeaturesIndicies = []
	removeIndicies = [0, 1, 2]
	delimiter = '\s+'
	if categoricalFeatureStrategy == 'onehot':
		X, y = oneHotReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies, removeIndicies=removeIndicies, delimiter=delimiter)
	elif categoricalFeatureStrategy == 'vdm':
		X, y = VDMReadInDS(datasetPath, classColumnIndex, categoricalFeaturesIndicies, removeIndicies=removeIndicies, delimiter=delimiter)
	return X, y

'''
Read in all datasets
'''
def getAllDS(categoricalFeatureStrategy):

	# Read in all datasets
	allDatasets = []

	X, y = importAnnealDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'anneal_' + categoricalFeatureStrategy))
	X, y = importAutoDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'auto_' + categoricalFeatureStrategy))
	X, y = importBalanceScaleDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'balanceScale_' + categoricalFeatureStrategy))
	X, y = importBreastTissueDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'breastTissue_' + categoricalFeatureStrategy))
	X, y = importCreditApprovalDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'creditApproval_' + categoricalFeatureStrategy))
	X, y = importDiabetesDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'diabetes_' + categoricalFeatureStrategy))
	X, y = importGermanCreditDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'germanCredit_' + categoricalFeatureStrategy))
	X, y = importGlassDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'glass_' + categoricalFeatureStrategy))
	X, y = importHeartDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'heart_' + categoricalFeatureStrategy))
	X, y = importHepatitisDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'hepatitis_' + categoricalFeatureStrategy))
	X, y = importImageSegmentationDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'imageSegmentation_' + categoricalFeatureStrategy))
	X, y = importIndianLiverDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'indianLiver_' + categoricalFeatureStrategy))
	X, y = importIonosphereDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'ionosphere_' + categoricalFeatureStrategy))
	X, y = importIrisDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'iris_' + categoricalFeatureStrategy))
	X, y = importLiverDisordersDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'liverDisorders_' + categoricalFeatureStrategy))
	X, y = importSonarDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'sonar_' + categoricalFeatureStrategy))
	X, y = importSoybeanDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'soybean_' + categoricalFeatureStrategy))
	X, y = importVehicleDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'vehicle_' + categoricalFeatureStrategy))
	X, y = importVoteDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'vote_' + categoricalFeatureStrategy))
	X, y = importVowelDS(categoricalFeatureStrategy)
	allDatasets.append((X, y, 'vowel_' + categoricalFeatureStrategy))

	return allDatasets


if __name__ == '__main__':

	#X, y = importAnnealDS()
	#X, y = importAutoDS()
	#X, y = importBalanceScaleDS()
	#X, y = importBreastTissueDS()
	#X, y = importCreditApprovalDS()
	#X, y = importDiabetesDS()
	#X, y = importGermanCreditDS()
	#X, y = importGlassDS()
	#X, y = importHeartDS()
	#X, y = importHepatitisDS()
	#X, y = importImageSegmentationDS()
	#X, y = importIndianLiverDS()
	#X, y = importIonosphereDS()
	#X, y = importIrisDS()
	#X, y = importLiverDisordersDS()
	#X, y = importSonarDS()
	#X, y = importSoybeanDS()
	#X, y = importVehicleDS()
	#X, y = importVoteDS()
	#X, y = importVowelDS()

	#print('')
	#print(X)
	#print('')
	#print(y)

	allDatasets = getAllDS('onehot')
	#allDatasets = getAllDS('vdm')

