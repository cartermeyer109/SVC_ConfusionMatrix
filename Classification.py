import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data
#This dataset is about 45k, so we will sample
df = pd.read_csv("loan_data.csv")

df.dropna(inplace=True) # Drops all indices with a null

#convert int64 to float64
df[df.select_dtypes('int64').columns] = df.select_dtypes('int64').astype('float64')

df = df.sample(10000) #Sample down dataset for run time

#Select variables
y = df["loan_status"].copy().to_numpy()
X = df[["credit_score", "person_age", "person_income", "loan_amnt", "cb_person_cred_hist_length",
        "person_emp_exp", "loan_int_rate", "loan_percent_income", ]].copy().to_numpy()

#Normalize the data
X = (X - np.average(X, axis=0)) / np.std(X, axis=0)

#Train/Test split with 30% left as a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Set model to a Support Vector Classifier using linear kernel
clf = SVC(C = 1.0, kernel='linear')

#Setting C to various numbers for training
#Returns "num" evenly spaced sampled from start to stop
parameters = {'C': np.linspace(0.1, 1.0, num=10)}

#Grid search uses k-fold validation to find the optimal parameters
grid_search = GridSearchCV(clf, param_grid=parameters, cv=5) #cv = k = 5
grid_search.fit(X_train, y_train)
score_clf = pd.DataFrame(grid_search.cv_results_)
print(score_clf[['param_C', 'mean_test_score', 'rank_test_score']])

#Get the best performing C
#nsmallest: bring best alpha to top row, iloc[0]: isolate top row, ['param_alpha']: get val from this col
optC = round((score_clf.nsmallest(1, 'rank_test_score')).iloc[0]['param_C'], 2)
clf.C = optC

print()
print(f"----Final Testing W/ Optimal C: {clf.C}----")
clf.fit(X_train, y_train) #A final train on the whole training set all at once
print(f"Score: {clf.score(X_test, y_test):.6f}")

#Print confusion Matrix
cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()