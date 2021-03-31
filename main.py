# Importing dependencies
import streamlit as st
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

#Importing all the models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# feature reduction method, projects feeatures to lower dimensional space
from sklearn.decomposition import PCA


favicon= "ParthRangarajanFavicon.ico"
#
st.set_page_config(page_title='WIB-EDA by ParthRangarajan', page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')

st.title("EDA with Breast Cancer, Wine and Iris datasets.")

st.header("""
 Explore different classifiers with different datasets.\n

""")

dataset_name=st.sidebar.selectbox("Select Dataset : ", ("Breast Cancer", "Iris", "Wine"))
st.write(dataset_name)

# different classifiers
classifier_name=st.sidebar.selectbox("Select Classifier : ", ("KNN", "SVM", "Random Forests"))

# input the dataset from sklearn
def get_data(data_name):
    if data_name=="Breast Cancer":
        data=datasets.load_breast_cancer()
    elif data_name=="Iris":
        data=datasets.load_iris()
    elif data_name=="Wine":
        data=datasets.load_wine()

    # spliting into X and y for train_test_split and ML model
    X=data.data
    y=data.target
    return X,y

X, y=get_data(dataset_name)
st.write("Shape : ",X.shape)
st.write("Number of classes : ", len(np.unique(y)))


def add_params(clf):
    params=dict()
    if clf=="KNN":
        knn=st.sidebar.slider("K (number of classes) : ", 1, 15)
        params["K"]=knn

    elif clf=="SVM":
        C=st.sidebar.slider("C (misclassification error avoidance) : ", 0.01, 10.0)
        params["C"]=C
    else:
        max_depth=st.sidebar.slider("Max Depth (for each tree): ", 2, 15)
        n_estimators=st.sidebar.slider("Number of estimators : ", 1, 100)
        params["max_depth"]=max_depth
        params["n_estimators"]=n_estimators
    return params

params=add_params(classifier_name)


# creating the classifiers
def get_classifier(clf, params):
    if clf=="KNN":
        clf=KNeighborsClassifier(n_neighbors=params["K"])
    elif clf=="SVM":
        clf=SVC(C=params["C"])
    else:
        clf=RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=1)
    return clf

clf=get_classifier(classifier_name,params)

# Classification
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=1)
# training classfier
clf.fit(X_train, y_train)
preds=clf.predict(X_test)
accuracy=accuracy_score(y_test, preds)

st.write(f"Classifier= {classifier_name}")
st.write(f"Accuracy= {round(accuracy,3)}%")

# Plotting Dataset
pca=PCA(2)
# unsupervised technique so doesn't require labels
X_proj=pca.fit_transform(X)

x1=X_proj[:, 0]
x2=X_proj[:, 1]

fig=plt.figure(figsize=(9,3))
plt.scatter(x1, x2, c=y, alpha=0.7, cmap='viridis')
plt.xlabel("Principal Component-1")
plt.ylabel("Principal Component-2")

plt.colorbar()

st.pyplot(fig)



st.write("\u00A9ParthRangarajan @2021")

st.write("[Github](https://github.com/parthrangarajan)      [Linkedln](https://www.linkedin.com/in/parthrangarajan/)        [Kaggle](https://www.kaggle.com/parthrangarajan01)      [Medium](https://medium.com/@parth.rangarajan2002)")


