# H2o-Review-Prediction-Engine 
:fire: A review sentiment prediction analysis for any dataset using h2o


# Classify Review Data 

This application helps you predict whether the review at hand will be positive or negative based on the dataset at hand. For example purposes, you can use this [UCI Machine Learning Datasets](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences)]

## A graphical interface to understand how your review will be classified


The code block contains both the ui elements and the underlying classification algorithm [In this case : h2o.ai](https://www.h2o.ai/).
Please note that the prediction depends purely on the supervised set input to the application


![Screenshot](/application.png "Application UI")


```python

#using tkinter for UI 
from tkinter import *
from tkinter import filedialog

#inititate the input file path and output
tsv_file=''


output_value=0.0


    
def get_pred():

    jobname=entry5.get()
    return jobname

#File :ask to open
def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    tsvfile=filename.get()
    print('Selected:', filename)

def AssignAction(event=None):
    filename = filedialog.askopenfilename()
    #tsvfile=filename 
    return filename

#prediction algorithm
def predict(job_title,w2v, gbm):
    import h2o
    words = tokenize(h2o.H2OFrame(job_title).ascharacter())
    job_title_vec = w2v.transform(words, aggregate_method="AVERAGE")
    #job_title_vec_2= job_title.cbind(job_title_vec)
    print(gbm.predict(job_title_vec))
    if gbm.predict(job_title_vec)>0.5:
        print("Most likely Negative")
    else:
        print("Probably positive")

        
def submitted():
    predx=get_pred()
    op=(predict([predx], w2v_model, gbm_model))
    print(predict([predx], w2v_model, gbm_model))

# tokenize to split each word of review
def tokenize(sentences, stop_word = ["ax","i","you","edu","s","t","m","subject","can","lines","re","what","there","all","we","one","the","a","an","of","or","in","for","by","on","but","is","in","a","not","with","as","was","if","they","are","this","and","it","have","from","at","my","be","by","not","that","to","from","com","org","like","likes","so"]
):
        tokenized = sentences.tokenize("\\W+")
        tokenized_lower = tokenized.tolower()
        tokenized_filtered = tokenized_lower[(tokenized_lower.nchar() >= 2) | (tokenized_lower.isna()),:]
        tokenized_words = tokenized_filtered[tokenized_filtered.grep("[0-9]",invert=True,output_logical=True),:]
        tokenized_words = tokenized_words[(tokenized_words.isna()) | (~ tokenized_words.isin(["ax","i","you","edu","s","t","m","subject","can","lines","re","what","there","all","we","one","the","a","an","of","or","in","for","by","on","but","is","in","a","not","with","as","was","if","they","are","this","and","it","have","from","at","my","be","by","not","that","to","from","com","org","like","likes","so"]
)),:]
        return tokenized_words


# The application interface
    
def analyze():
    

    tsv_file=AssignAction()


    import time
    time.sleep(10) 

    import h2o

    h2o.init(max_mem_size="2G")

    label3 = Label(root, text="Building Model")
    label3.pack()

    import pandas as pd
    import numpy as np

    csv_table=pd.read_table(tsv_file,sep='\t',header=None,names = ["Review", "Response"] )
    csv_table.to_csv('reviews.csv',index=False)

#print(type(csv_table))
    csv_table
    reviews = h2o.h2o.H2OFrame(csv_table)


    words = tokenize(reviews["Review"])


    from h2o.estimators.word2vec import H2OWord2vecEstimator


    w2v_model = H2OWord2vecEstimator(vec_size = 100, model_id = "w2v.hex")
    w2v_model.train(training_frame=words)

    review_vecs = w2v_model.transform(words, aggregate_method = "AVERAGE")

#ext_reviews

    ext_reviews = reviews.cbind(review_vecs)

    data_split = ext_reviews.split_frame(ratios=[0.8])

    ext_train = data_split[0]
    ext_test = data_split[1]


    #print("Build a basic GBM model")
    gbm_model = h2o.estimators.gbm.H2OGradientBoostingEstimator()
    gbm_model.train(x = ext_reviews.names,
                    y="Response", 
                    training_frame = data_split[0], 
                    validation_frame = data_split[1])
    #output_value=predict([entry5.get()], w2v_model, gbm_model)
    #label7 = Label(root, text=str(output_value))
    #label7.pack()
    x=entry5.get()
    y=h2o.H2OFrame([x])
    y.col_names = ['Review']
    return(predict([y], w2v_model, gbm_model))


#root.mainloop()
    
root = Tk()
root.title("h2o Data Analytics")


#Label 1
label1 = Label(root,text = 'Prediction of Comments')
label1.pack()
label1.config(justify = CENTER)


label5 = Label(root, text="Enter a comment that you want to predict")
label5.pack()
#label1.config(justify = CENTER)

entry5 = Entry(root, width = 100)
entry5.pack()



label4 = Label(root, text="Upload review data file to start the analysis based on the data")
label4.pack()

button6 = Button(root, text = 'Start Now')
button6.pack() 
button6.config(command = analyze)




root.mainloop()

```

    Checking whether there is an H2O instance running at http://localhost:54321. connected.



<div style="overflow:auto"><table style="width:50%"><tr><td>H2O cluster uptime:</td>
<td>4 hours 52 mins</td></tr>
<tr><td>H2O cluster timezone:</td>
<td>America/New_York</td></tr>
<tr><td>H2O data parsing timezone:</td>
<td>UTC</td></tr>
<tr><td>H2O cluster version:</td>
<td>3.22.0.1</td></tr>
<tr><td>H2O cluster version age:</td>
<td>15 days </td></tr>
<tr><td>H2O cluster name:</td>
<td>H2O_from_python_narensham_b4de52</td></tr>
<tr><td>H2O cluster total nodes:</td>
<td>1</td></tr>
<tr><td>H2O cluster free memory:</td>
<td>1.472 Gb</td></tr>
<tr><td>H2O cluster total cores:</td>
<td>8</td></tr>
<tr><td>H2O cluster allowed cores:</td>
<td>8</td></tr>
<tr><td>H2O cluster status:</td>
<td>locked, healthy</td></tr>
<tr><td>H2O connection url:</td>
<td>http://localhost:54321</td></tr>
<tr><td>H2O connection proxy:</td>
<td>None</td></tr>
<tr><td>H2O internal security:</td>
<td>False</td></tr>
<tr><td>H2O API Extensions:</td>
<td>XGBoost, Algos, AutoML, Core V3, Core V4</td></tr>
<tr><td>Python version:</td>
<td>3.6.4 final</td></tr></table></div>


    Parse progress: |█████████████████████████████████████████████████████████| 100%
    word2vec Model Build progress: |██████████████████████████████████████████| 100%
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%



<table>
<thead>
<tr><th>Review                            </th></tr>
</thead>
<tbody>
<tr><td>What is this about, I have no clue</td></tr>
</tbody>
</table>


    Parse progress: |█████████████████████████████████████████████████████████| 100%
    gbm prediction progress: |████████████████████████████████████████████████| 100%



<table>
<thead>
<tr><th style="text-align: right;">  predict</th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;"> 0.338029</td></tr>
</tbody>
</table>


    
    gbm prediction progress: |████████████████████████████████████████████████| 100%
    Most likely Negative



```python
The above statement specifies the likely classification based on the dataset provided.
```
