import streamlit as st
st.title("IRIS API")

# slider
# age=st.slider("How old are you?",0,130,25)
# st.write("I'm",age,"years old.")

# webpages
sl=st.slider("Sepal Length",4.3,7.9,5.1)
sw=st.slider("Sepal Width",3.5,4.5,2.2)
pl=st.slider("Petal Length",1.0,6.9,5.5)
pw=st.slider("Petal Width",0.1,2.5,2.2)

# model
from sklearn.datasets import load_iris
iris=load_iris()

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(iris.data,iris.target)
pred=model.predict([[sl,sw,pl,pw]])
pred=iris.target_names[pred[0]]
st.title(f"The flower species is {pred}")