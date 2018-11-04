import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt


def fit_and_score(degree, train_test, C=1.0):
    X_train, X_test, y_train, y_test = train_test
    model = make_pipeline(StandardScaler(),
                          PolynomialFeatures(degree=degree),
                          LogisticRegression(solver='lbfgs', C=C))
    model.fit(X_train, y_train)
    report = classification_report(y_true=y_test,
                                   y_pred=model.predict(X_test))
    return model, report


def pred_grid(model, x_min, x_max, y_min, y_max, n):
    x, y = np.meshgrid(np.linspace(x_min, x_max, n),
                       np.linspace(y_min, y_max, n))
    xy_flat = np.vstack([x.flat, y.flat]).T
    scores = model.predict_proba(xy_flat)[:, 1].squeeze()
    return x, y, scores.reshape(x.shape)


def show_pred(model, X, y, lrbt, n):
    x_grid, y_grid, scores = pred_grid(model, *lrbt, n)
    fig = plt.figure(figsize=(15, 15))
    ax = plt.subplot(111)
    ax.contourf(x_grid, y_grid, scores, levels=10, cmap='RdBu', alpha=0.7)
    ax.scatter(*X[np.where(y == 1)].T, c='b', label='+1')
    ax.scatter(*X[np.where(y == 0)].T, c='r', label='-1')
