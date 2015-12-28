from subprocess import check_call
import sklearn.tree as tree
import xgboost as xgb
import seaborn as sns


def printDecisionTree(fn, decisionTree, featureNames=None, opClassNames=None):
    """
    Creates a pdf for the given Decision Tree from sklearn.tree package

    :param fn: filename
    :param decisionTree: decision tree object
    :param featureNames (Optional, Default = None) : list of features used by the decision tree
    :param opClassNames (Optional, Default = None) : list of output class names
    """
    with open(fn + ".dot", "w") as file:
        tree.export_graphviz(decisionTree, out_file=file,feature_names=featureNames, class_names=opClassNames, filled=True, rounded=True)

    check_call(["dot", "-Tpdf", fn + ".dot", "-o", fn + ".pdf"])

def printXGBoostTree(fn, xgBoostTree, numTrees=2, yesColor='#0000FF', noColor='#FF0000'):
    """
    Creates a pdf for the given XGBoost Tree from xgboost package

    :param fn: filename
    :param xgBoostTree: XGBoost tree object
    :param numTrees (Optional, Default = 2) : Number of decision trees to draw
    :param yesColor (Optional, Default = '#0000FF') : Color of correct output classes
    :param noColor (Optional, Default = '#FF0000'): Color of wrong output classes
    """
    with open(fn + ".dot", "w") as file:
        val = xgb.to_graphviz(xgBoostTree, num_trees=numTrees, yes_color=yesColor, no_color=noColor)

    val.save(fn + ".dot")
    check_call(["dot", "-Tpdf", fn + ".dot", "-o", fn + ".pdf"])


def printFeatureImportances(featurenames, featureImportances):
    """
    Prints the feature importances of classifiers or regressors in the sklearn package

    :param featurenames: list of feature names
    :param featureImportances: list of feature importance values eg. decisionTree.feature_importances_
    """
    featureImportances = [(feature, importance) for feature, importance in zip(featurenames, featureImportances)]
    featureImportances = sorted(featureImportances, key=lambda x: x[1], reverse=True)
    print("Feature Importances : \n", featureImportances)

def printXGBFeatureImportances(featurenames, xgbTree):
    featureNames = featurenames
    featureImportances = [(feature, importance)
                          for feature, importance in zip(featureNames, sorted(xgbTree.booster().get_fscore(), key=lambda x: x[1]))]
    print("Feature Importances : \n", featureImportances)
    featureImportance = xgb.plot_importance(xgbTree)
    sns.plt.show()


