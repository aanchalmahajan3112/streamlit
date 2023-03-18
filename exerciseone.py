import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import plotly.graph_objs as go
import plotly.offline as py
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

##set title

st.title('Total Data Science')
image=Image.open(r'C:\Users\yugan\Downloads\maha-shivratri-lord-shiva-trishul-illustration-background-vector-140771911.jpg')
st.image(image,use_column_width=True)

#set subtitle
st.write("""
	 # A simple Data App with Streamlit""")


st.write('''
	### Let's Explore diff classifiers and datasets''' )

dataset_name=st.sidebar.selectbox('Select dataset',('Breast Cancer','Iris','Wine'))
classifier_name=st.sidebar.selectbox('Select Classifier',('SVM','KNN'))

def get_dataset(name):
	data=None
	if name=='Iris':
		data=datasets.load_iris()
	elif name=="Wine":
		data=datasets.load_wine()
	else:
		data=datasets.load_breast_cancer()
	x=data.data
	y=data.target

	return x,y

x,y=get_dataset(dataset_name)
st.dataframe(x)
st.write('Shape of ur dataset is :',x.shape)
st.write('Unique target variables:',len(np.unique(y)))


fig=plt.figure()
sns.boxplot(data=x,orient='h')
st.pyplot()


plt.hist(x)
st.pyplot()

#building our algorithm
def add_parameter(name_of_clf):
	params=dict()
	if name_of_clf=='SVM':
		C=st.sidebar.slider('C',0.01,15.0)
		params['C']=C
	else:
		 name_of_clf=='KNN'
		 k=st.sidebar.slider('K',1,15)
		 params['k']=k
	return params

params=add_parameter(classifier_name )

#accessing our classifier

def get_classifier(name_of_clf,params):
	clf=None
	if name_of_clf=='SVM':
		clf=SVC(C=params['C'])
	elif name_of_clf=='KNN':
		clf=KNeighborsClassifier(n_neighbors=params['k'])
	else:
		st.warning('U did nt select any option, please select one')

	return clf

clf=get_classifier(classifier_name,params)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=10)

clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
st.write(y_pred)

accuracy=accuracy_score(y_test,y_pred)

st.write('classifier_name:',classifier_name)
st.write('Accuracy for ur model is:',accuracy)
