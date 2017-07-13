import numpy as np
from lightfm import LightFM
from fetch_lastfm import fetch_lastfm
from lightfm.evaluation import auc_score
from rocForLossFunctions import plot_roc

# fetching and parsing dataset from lastfm
data = fetch_lastfm()

loss_function=['warp','bpr','logistic']

print('Available loss functions: ')
print(' \t 1 - warp \t 2 - bpr \t 3 - logistic ')
print '\n'
auc_scores=[]

print('Finding optimum loss function based on AUC score ...')

# Finding optimum loss function
for x in range(len(loss_function)):
    model = LightFM(loss=loss_function[x])
    model.fit(data['matrix'], epochs=30, num_threads=2)
    auc_scores.append(auc_score(model,data['matrix']).mean())

optimumloss=loss_function[auc_scores.index(max(auc_scores))]

print('Maximum AUC is : %.2f for -- %s loss function (refer ROC plot) \n' % (max(auc_scores),optimumloss))

# creating the model using optimum loss function
model = LightFM(loss=optimumloss)

# training the model
model.fit(data['matrix'], epochs=30, num_threads=2)


# ROC plot
plot_roc(data)

# Get recommendationns function
def get_recommendations(model, coo_mtrx, users_ids):

    n_items = coo_mtrx.shape[1]

    for user in users_ids:

        # TODO create known positives
        # Artists the model predicts they will like
        scores = model.predict(user, np.arange(n_items))
        top_scores = np.argsort(-scores)[:3]

        print 'Recomendations for user %s:' % user

        for x in top_scores.tolist():
            for artist, values in data['artists'].iteritems():
                if int(x) == values['id']:
                    print '   - %s' % values['name']

        print '\n' # Get it pretty


user_1 = raw_input('Select user_1 (0 to %s): ' % data['users'])
user_2 = raw_input('Select user_2 (0 to %s): ' % data['users'])
user_3 = raw_input('Select user_3 (0 to %s): ' % data['users'])
print '\n' # Get it pretty

get_recommendations(model, data['matrix'], [user_1, user_2, user_3])

