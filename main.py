import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import sklearn
import joblib
import json

## Side Tab:
l=["Introduction","Predict your Credit Score"]
st.sidebar.subheader("Here's what you can do:")
option=st.sidebar.selectbox("Choose what you want to do:",l)

def page_1():
    ## Intro Tab::
    image = Image.open('Credit_Risk.jpg')

    ## Displaying the image:
    st.image(image,use_column_width="always")

    ## Headers:
    st.title("Welcome to this Mock Credit Risk Simulator")
    st.header("Here's the drill. You get me whatever I need and I predict whether you are eligible or not. DEAL!")
    st.subheader("Let's get started...")

def page_2():
    data={}
    ## Details Tab:
    st.header("Gimme your details and I will deliver magic!")

    #Full Name:
    first,last=st.columns(2)
    first=first.text_input("Enter your First Name:")
    last=last.text_input("Enter your Last Name:")
    data["First Name"]=first
    data["Last Name"]=last

    name=first+" "+last
    data["Full Name"]=name

    ##Age:
    age=st.slider("Enter your Age:",10,70)
    data["Age"]=age

    ##Annual Income:
    ai=st.number_input("Enter your Annual Income:",1000,100000)
    data["Annual Income"]=ai

    ##Home Ownership:
    ho=st.selectbox("What is the type of House Ownership:", ["RENT", "OWN", "MORTGAGE","OTHER"])
    data["Home Ownership"]=ho

    ##Employment Length:
    el=st.number_input("Enter your Work Experience in years:",2,50)
    data["Employment Length"]=el

    ##Loan Intent:
    li=st.selectbox("Why do you want a loan?", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION',
                                                'HOMEIMPROVEMENT'])
    data["Loan Intent"]=li

    ##Loan Grade:
    lg=st.selectbox("Grade of Loan expected?", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    data["Loan Grade"]=lg

    ## Loan Amount:
    la=st.number_input("Enter your Work Experience in years:",100,50000)
    data["Loan Amount"]=la

    ## loan_percent_income:
    lpi=st.number_input("Enter your % Income to be used for repaying:",0,100)
    data["Loan Percent Income"]=lpi

    ## cb_person_default_on_file:
    def_his=st.selectbox("Have your ever defaulted?",["Y","N"]) 
    data["Previous Defaults"]=def_his

    ## cb_person_cred_hist_length:
    n_def=st.slider("Total Number of Defaults:",0,50)
    data["Number of Defaults"]=n_def

    ## Make a submit button:
    data_display=json.dumps(data)
    temp=pd.DataFrame(data,index=[0])  ## making a record

    ## Display the input data as a json:
    if st.button("Display Data",key = 8)==1:
        st.write("The data in JSON Format:")
        st.write(data_display)        
        st.write("\nThe data in Tabular Format:")
        st.write(temp)   
 
    ## Display the prediction:
    if st.button("Predict Credit Score",key = 9)==1:
        ## Order of passing the data into the pipeline:
        cols=['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
       'loan_percent_income', 'cb_person_cred_hist_length',
       'person_home_ownership', 'loan_intent', 'loan_grade',
       'cb_person_default_on_file']  ## List of columns of the original dataframe
                
        input_data=[[data["Age"],data["Annual Income"],data["Employment Length"],data["Loan Amount"],
                     round(data["Loan Percent Income"]/100,2),data["Number of Defaults"],
                     data["Home Ownership"],data["Loan Intent"],data["Loan Grade"],data["Previous Defaults"]]]
        
        pipe=joblib.load('best_pipeline.pkl')  ## Loading the pipeline
        
        input_data=pd.DataFrame(input_data,columns=cols)  ## Converting input into a dataframe with respective columns

        res=pipe.predict(input_data)[0]  ## Predicting the class
        out={1:"The Customer is capable of DEFAULTING. Hence it is RISKY to provide loan!", 0:"The Customer is capable of NOT DEFAULTING. Hence it is POSSIBLE to provide loan!"}
        st.write(f"The Final Verdict obtained from the given model is that : {out[res]}")
        

if option==l[0]:
    page_1()

if option==l[1]:
    page_2()



