import numpy as np
import matplotlib.pyplot as plt 
from lightfm import LightFM 
from lightfm.evaluation import auc_score


def plot_roc(data):
	alpha = 1e-05
	epochs = 50
	num_components = 32

	warp_model = LightFM(no_components=num_components,
	                    loss='warp',
	                    learning_schedule='adagrad',
	                    max_sampled=3,
	                    user_alpha=alpha,
	                    item_alpha=alpha)

	bpr_model = LightFM(no_components=num_components,
	                    loss='bpr',
	                    learning_schedule='adagrad',
	                    user_alpha=alpha,
	                    item_alpha=alpha)

	logistic_model=LightFM(no_components=num_components,
	                    loss='logistic',
	                    learning_schedule='adagrad',
	                    user_alpha=alpha,
	                    item_alpha=alpha)

	warp_auc = []
	bpr_auc = []
	logistic_auc = []

	for epoch in range(epochs):
	    warp_model.fit_partial(data['matrix'], epochs=5)
	    warp_auc.append(auc_score(warp_model, data['matrix']).mean())
	    
	for epoch in range(epochs):
	    bpr_model.fit_partial(data['matrix'], epochs=5)
	    bpr_auc.append(auc_score(bpr_model, data['matrix']).mean())

	for epoch in range(epochs):
	    logistic_model.fit_partial(data['matrix'], epochs=5)
	    logistic_auc.append(auc_score(bpr_model, data['matrix']).mean())

	x = np.arange(epochs)
	plt.plot(x, np.array(warp_auc))
	plt.plot(x, np.array(bpr_auc))
	plt.plot(x, np.array(logistic_auc))
	plt.legend(['WARP AUC', 'BPR AUC', 'LOGISTIC AUC'], loc='upper right')
	
	return plt.show(block=False)