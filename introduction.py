import streamlit as st

st.title("Our First Streamlit App")

from PIL import Image
st.subheader('This is a subheader')
image=Image.open(r'C:\Users\yugan\Downloads\maha-shivratri-lord-shiva-trishul-illustration-background-vector-140771911.jpg')
st.image(image,use_column_width=True)
st.write('Om namah shivay')
st.markdown('this is a markdown cell')

st.success('Congrats to run it successfully')
st.info('this is an informaton for u ')
st.warning('Be cautious')
st.error('Oops you run into an error')
st.help(range)

import numpy as np
import pandas as pd
dataframe=np.random.rand(10,20)
st.dataframe(dataframe)
st.text("---"*100)

df=pd.DataFrame(np.random.rand(10,20),columns=('col %d' % i for i in range(20)))
st.dataframe(df.style.highlight_max(axis=1))

st.text("---"*100)

#Display chart

chart_data=pd.DataFrame(np.random.randn(20,3),columns=['a','b','c'])
st.line_chart(chart_data)

st.text("---"*100)

st.area_chart(chart_data)

chart_data=pd.DataFrame(np.random.randn(50,3),columns=['a','b','c'])
st.bar_chart(chart_data)

import matplotlib.pyplot as plt
arr=np.random.normal(1,1,size=100)
plt.hist(arr,bins=20)
st.pyplot()

st.text("---"*100)

import plotly
import  plotly.figure_factory as ff


#adding distplot
x1=np.random.randn(200)-2
x2=np.random.randn(200)
x3=np.random.randn(200)-2

hist_data=[x1,x2,x3]
group_labels=['Group1','Group2','Group3']
fig=ff.create_distplot(hist_data,group_labels,bin_size=[.2,.25,.5])
st.plotly_chart(fig,use_container_width=True)

st.text('---'*100)

df=pd.DataFrame(np.random.randn(100,2)/[50,50]+[37.76,-122.4],columns=['lat','lon'])
st.map(df)
st.text('---'*100)

#creating buttons
if st.button('Say hello'):
	st.write("hi there")

else:
	st.write('why r u here')

st.text('---'*100)

genre=st.radio('What is your favourite genre',('Comedy','Drama','Documentary'))

if genre=='Comedy':
	st.write("oh u like comedy")

elif genre=="Drama":
	st.write('yeah drama is cool')
else:
	st.write('I see')

st.text('---'*100)

#select button
option=st.selectbox('How was ur night',('Fantastic','Awesome','So-so'))
st.write("You said ur night was:",option)

st.text('---'*100)

option=st.multiselect('How was ur night, u can select multiple choice',('Fantastic','Awesome','So-so'))
st.write("You said ur night was:",option)

st.text('---'*100)

age=st.slider('How old r u ',0,150,10)
st.write('Your age is :',age)

st.text('---'*100)

values=st.slider('Select a range of values',0,200,(15,80))
st.write('You selected a range between:',values)

st.text('---'*100)

number=st.number_input("Input number")
st.write('the no u inputed is:',number)

st.text('---'*100)
st.text('---'*100)



#file uploader
upload_file=st.file_uploader('Choose a csv file',type='csv')

if upload_file is not None:
	data=pd.read_csv(upload_file)
	st.write(data)
	st.success('successfully uploaded')
else:
	st.markdown('Please upload a csv file.')

st.text('---'*100)

#color picker
color = st.color_picker('Pick A Color', '#00f900')
st.write('The current color is', color)

st.text('---'*100)
st.text('---'*100)

#side bar

add_sidebar=st.sidebar.selectbox('what is ur fav course',('TDS','Ineuron','others'))

import time 
my_bar=st.progress(0)
for percent_complete in range(100):
	time.sleep(0.1)
	my_bar.progress(percent_complete+1)

st.text('---'*100)
st.text('---'*100)

with st.spinner('wait for it ......'):
	time.sleep(5)
st.success('successfully')

st.balloons()