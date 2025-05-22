import streamlit as st
import pickle
import sklearn
st.header("Titanic Survival Prediction")
st.subheader("Predicting Survival on the Titanic")

st.image("titanic.jpeg")
st.text('''The RMS Titanic was a British ocean liner that sank on its maiden voyage from Southampton, England to New York City on April 14-15, 1912, after striking an iceberg. The ship was built by Harland and Wolff in Belfast and was owned by the White Star Line. The sinking of the Titanic resulted in the deaths of about 1,500 passengers and crew.''')
model=pickle.load(open('model.pkl','rb'))
l_sex=pickle.load(open('l_sex.pkl','rb'))
l_emb=pickle.load(open('l_emb.pkl','rb'))

pclass=st.number_input("Passenger Class")
#pclass=st.radio("Select Passenger Class",(1,2,3)) -another mthd
sex=st.text_input("enter sex:[male,female]")
age=st.number_input("Age")
sibsp=st.number_input("Number of Siblings/Spouses Abroad")
parch=st.number_input("Number of Parents/Children abroad")
fare=st.number_input("Fare")
embarked=st.text_input("Embarked:[S,C,Q]")
if st.button("Predict"):
    sex_l=l_sex.transform([sex])[0]
    embarked_l=l_emb.transform([embarked])[0]
    predict=model.predict([[pclass,sex_l,age,sibsp,parch,fare,embarked_l]])[0]
    if predict==1:
        st.success('Survived')
    else:
        st.warning('Did not survive')