
# coding: utf-8

# In[ ]:



# using tkinter for UI 
from tkinter import *
from tkinter import filedialog

#inititate the input file path and output
tsv_file=''


output_value=0.0


    
def get_pred():

    jobname=entry5.get()
    return jobname


def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    tsvfile=filename.get()
    print('Selected:', filename)
    
def AssignAction(event=None):
    filename = filedialog.askopenfilename()
    #tsvfile=filename
    return filename


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


