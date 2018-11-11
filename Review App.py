
# coding: utf-8

# In[1]:


import h2o


# In[2]:


h2o.init(max_mem_size="2G")


# In[123]:


docker_data_path = "/Users/narensham/Downloads/sentiment labelled sentences/yelp_labelled.txt"


# In[124]:


docker_data_path


# In[136]:


import pandas as pd
import numpy as np

tsv_file='/Users/narensham/Downloads/sentiment labelled sentences/yelp_labelled.txt'
csv_table=pd.read_table(tsv_file,sep='\t',header=None,names = ["Review", "Response"] )
csv_table.to_csv('reviews.csv',index=False)

print(type(csv_table))
csv_table
reviews = h2o.h2o.H2OFrame(csv_table)



# In[128]:


reviews


# In[137]:


reviews["Response"].table()


# In[138]:


STOP_WORDS = ["ax","i","you","edu","s","t","m","subject","can","lines","re","what",
               "there","all","we","one","the","a","an","of","or","in","for","by","on",
               "but","is","in","a","not","with","as","was","if","they","are","this","and","it","have",
               "from","at","my","be","by","not","that","to","from","com","org","like","likes","so"]


# In[134]:



def tokenize(sentences, stop_word = STOP_WORDS):
    tokenized = sentences.tokenize("\\W+")
    tokenized_lower = tokenized.tolower()
    tokenized_filtered = tokenized_lower[(tokenized_lower.nchar() >= 2) | (tokenized_lower.isna()),:]
    tokenized_words = tokenized_filtered[tokenized_filtered.grep("[0-9]",invert=True,output_logical=True),:]
    tokenized_words = tokenized_words[(tokenized_words.isna()) | (~ tokenized_words.isin(STOP_WORDS)),:]
    return tokenized_words


# In[140]:


words = tokenize(reviews["Review"])


# In[141]:


words.head()


# In[142]:


from h2o.estimators.word2vec import H2OWord2vecEstimator


w2v_model = H2OWord2vecEstimator(vec_size = 100, model_id = "w2v.hex")
w2v_model.train(training_frame=words)


# In[148]:


w2v_model.find_synonyms("good", count = 5)


# In[149]:


review_vecs = w2v_model.transform(words, aggregate_method = "AVERAGE")


# In[150]:


ext_reviews = reviews.cbind(review_vecs)
ext_reviews


# In[151]:


data_split = ext_reviews.split_frame(ratios=[0.8])

ext_train = data_split[0]
ext_test = data_split[1]



# In[152]:





print("Build a basic GBM model")
gbm_model = H2OGradientBoostingEstimator()
gbm_model.train(x = ext_reviews.names,
                y="Response", 
                training_frame = data_split[0], 
                validation_frame = data_split[1])


# In[153]:


def predict(job_title,w2v, gbm):
    words = tokenize(h2o.H2OFrame(job_title).ascharacter())
    job_title_vec = w2v.transform(words, aggregate_method="AVERAGE")
    print(gbm.predict(test_data=job_title_vec))


# In[157]:


print(predict(["a huge selection"], w2v_model, gbm_model))
print(predict(["a little"], w2v_model, gbm_model))


# In[158]:


plt=gbm_model.varimp_plot()


# In[159]:


h2o.cluster().shutdown()

