import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

RUNS = 10

def read_dataset(path):
	dataset = pd.read_csv(path).values[:, ]
	X, y = dataset[:, :-1], dataset[:, -1]
	print('Dataset shape: ', X.shape, y.shape)
	print('Number of classes: ', np.unique(y).shape[0])
	return X, y

def sample_batch(size, p, method='entropy'):
	'''Returns indices of samples'''
	if method == 'entropy':
		entr = (- p * np.log2(p)).sum(axis=1)
		return np.argsort(entr)[::-1][:size]
	elif method == 'margin_sampling':
		pass
	elif method == 'random':
		return np.random.choice(np.arange(p.shape[0]), size=size, replace=False)

if __name__ == '__main__':

	dataset_path = sys.argv[1]
	X, y = read_dataset(dataset_path)
    # imputer = SimpleImputer(missing_values = 0, strategy ="mean")
    # imputer = imputer.fit(X)
    # imputer = imputer.transorm(X)

	scaler = StandardScaler()
	scaler = scaler.fit(X)
	X = scaler.transform(X)

	X_train, X_test, y_train, y_test = train_test_split(X, y, 
										test_size=0.4, random_state=21)
	TRAIN_SIZE = X_train.shape[0]
	TEST_SIZE = X_test.shape[0]
	INIT_SIZE = 100
	BATCH_SIZE = 100

	for _ in range(RUNS):

		# Active Learning
	
		# Generate initial train set by sampling randomly
		perm = np.random.choice(np.arange(TRAIN_SIZE),
				size=INIT_SIZE, replace=False)
		X_labelled = np.copy(X_train[perm, :])
		y_labelled = np.copy(y_train[perm])

		X_unlabelled = np.delete(X_train, perm, axis=0)
		y_unlabelled = np.delete(y_train, perm, axis=0)

		active_learn_acc = []
		instances = []
		while X_labelled.shape[0] < TRAIN_SIZE:
			model = LogisticRegression(solver='lbfgs', multi_class='multinomial',
				random_state=21).fit(X_labelled, y_labelled)
			active_learn_acc.append(model.score(X_test, y_test))
			instances.append(X_labelled.shape[0])

			# Calculate probabilities for each sample for each class
			probs = model.predict_proba(X_unlabelled)

			# Add the most uncertain samples to training set
			perm = sample_batch(min(BATCH_SIZE, probs.shape[0]), probs, 'entropy')
			X_labelled = np.append(X_labelled, X_unlabelled[perm, :], axis=0)
			y_labelled = np.append(y_labelled, y_unlabelled[perm], axis=0)
			X_unlabelled = np.delete(X_unlabelled, perm, axis=0)
			y_unlabelled = np.delete(y_unlabelled, perm, axis=0)


		# Passive Learning

		# Generate initial train set by sampling randomly
		perm = np.random.choice(np.arange(TRAIN_SIZE), size=INIT_SIZE, replace=False)
		X_labelled = np.copy(X_train[perm, :])
		y_labelled = np.copy(y_train[perm])

		X_unlabelled = np.delete(X_train, perm, axis=0)
		y_unlabelled = np.delete(y_train, perm, axis=0)

		passive_learn_acc = []
		while X_labelled.shape[0] < TRAIN_SIZE:
			model = LogisticRegression(solver='lbfgs', multi_class='multinomial',
				random_state=21).fit(X_labelled, y_labelled)
			passive_learn_acc.append(model.score(X_test, y_test))	

			# Calculate probabilities for each sample for each class
			probs = model.predict_proba(X_unlabelled)

			# Add the most uncertain samples to training set
			perm = sample_batch(min(BATCH_SIZE, probs.shape[0]), probs, 'random')
			X_labelled = np.append(X_labelled, X_unlabelled[perm, :], axis=0)
			y_labelled = np.append(y_labelled, y_unlabelled[perm], axis=0)
			X_unlabelled = np.delete(X_unlabelled, perm, axis=0)
			y_unlabelled = np.delete(y_unlabelled, perm, axis=0)

		plt.plot(instances, passive_learn_acc, 'r--', label='Passive Learning')
		plt.plot(instances, active_learn_acc, 'g-', label='Active Learning')
		plt.title('Model:Logistic Regression        Dataset: waveform + noise')
		plt.xlabel('Number of instances')
		plt.ylabel('Accuracy score')
		plt.legend()
		plt.show()
