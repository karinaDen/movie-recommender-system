from surprise import Reader, Dataset, KNNBasic, SVD, NMF
from surprise.model_selection import GridSearchCV, cross_validate, KFold


def get_cross_validation(algo, data, cv=3):
    cross_validate(algo=algo, data=data, measures=['RMSE'], cv=3, verbose=True)
