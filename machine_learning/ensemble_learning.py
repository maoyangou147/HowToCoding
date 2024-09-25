from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.datasets import load_iris

from sklearn.metrics import accuracy_score

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)

    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)

    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, x2, y_pred, cmap=custom_cmap, alpha=0.3)

    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)

    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'yo')
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 1], 'bs')
    plt.axis(axes)
    plt.xlabel('x1')
    plt.ylabel('x2')

def hard_vote():
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'yo')
    # plt.plot(X[:, 0][y == 0], X[:, 1][y == 1], 'bs')

    # plt.show()

    log_clf = LogisticRegression(random_state=42)
    rnd_clf = RandomForestClassifier(random_state=42)
    svm_clf = SVC(random_state=42)

    voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')
    voting_clf.fit(X_train, y_train)

    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

def soft_vote():
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'yo')
    # plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')

    # plt.show()

    log_clf = LogisticRegression(random_state=42)
    rnd_clf = RandomForestClassifier(random_state=42)
    svm_clf = SVC(random_state=42, probability=True)

    voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='soft')
    voting_clf.fit(X_train, y_train)

    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

def bagging():
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(),
        n_estimators = 500, 
        max_samples=100,
        bootstrap=True,
        n_jobs=-1, 
        random_state=42
    )
    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)

    print(bag_clf.__class__.__name__, accuracy_score(y_test, y_pred))

    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    y_pred = tree_clf.predict(X_test)

    print(tree_clf.__class__.__name__, accuracy_score(y_test, y_pred))

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plot_decision_boundary(tree_clf, X, y)
    plt.title('Decision Tree')
    plt.subplot(122)
    plot_decision_boundary(bag_clf, X, y)
    plt.title('Decision Tree With Bagging')

    plt.show()

def oob():
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(),
        n_estimators = 500, 
        max_samples=100,
        bootstrap=True,
        n_jobs=-1, 
        random_state=42,
        oob_score=True
    )
    bag_clf.fit(X_train, y_train)
    print(bag_clf.oob_score_)

def random_forest():
    iris = load_iris()
    rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    rf_clf.fit(iris['data'], iris['target'])

if __name__ == '__main__':
    # 硬投票实验
    # hard_vote()
    # 软投票实验
    # soft_vote()

    bagging()

    




