from surprise import Reader, Dataset, KNNBasic, SVD, NMF
from surprise.model_selection import GridSearchCV, cross_validate, KFold

def find_best_params(data, param_grid, algo=KNNBasic):
 
    gs = GridSearchCV(algo, measures=['RMSE'], param_grid=param_grid)
    gs.fit(data)
    return gs.best_score['rmse'], gs.best_params['rmse']



