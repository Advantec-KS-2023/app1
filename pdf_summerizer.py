#!/usr/bin/env python
# coding: utf-8

# In[11]:


import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from PyPDF2 import PdfReader
#import nltk
#from nltk.tokenize import word_tokenize
import os, torch, gc, requests, re
#from bs4 import BeautifulSoup


# In[3]:

#nltk.download('punkt')
#try:
#    nltk.data.find('tokenizers/punkt')
#except LookupError:
#    nltk.download('punkt')


# In[5]:


MODEL_PATH = "tsmatz/mt5_summarize_japanese"
#MODEL_PATH = 'google/mt5-large'


# In[7]:


#tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
seq2seq = pipeline("summarization", model = MODEL_PATH, min_length=50)

# In[9]:
def extract_the_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.content
    except requests.exceptions.RequestException as e:
        st.error(f'An error occured while fetching the paper: {e}')

    soup = BeautifulSoup(html_content, 'html.parser')
    for script in soup(['script', 'style']):
        script.decompose()
    text= soup.get_text()
    text = text.strip()
    return text

def extract_the_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = " "
    for page in reader.pages:
        cs = page.extract_text()
        cs2 = cs.replace('\n', ' ')
        text += cs2
    return text.strip()


# In[13]:
gc.collect()

def summarize_text(text):
    tokens = word_tokenize(text)
    input = tokenizer(tokens, return_tensors='pt', truncation=True, padding = True)

    with torch.no_grad():
        output = model(**input)

    probs = torch.nn.functional.softmax(output.logits, dim=-1)

    top_sentence =[]
    for i in probs[0].topk(3):
        top_sentence.append(tokens[i])
    summary = " ".join(top_sentence)
    return summary.strip()


# In[15]:


st.title('PDF summarize')


# In[17]:


upload_file = st.file_uploader('Choose a PDF file', type='pdf')

                         

# In[19]:


if upload_file is not None:
    pdf_content = upload_file.getvalue()
    pdf_path = 'temp.pdf'
    with open(pdf_path, "wb") as f:
        f.write(pdf_content)

    st.subheader('Extract Text: ')
    extractes_text = extract_the_text_from_pdf(pdf_path)
    #extractes_text = extract_the_text_from_url(url_name)
    st.write(extractes_text[:1000]+"....")

    st.subheader('Summary:')
    summary = seq2seq(extractes_text)[0]
    st.write(summary.get('summary_text'))

    os.remove(pdf_path)
    gc.collect()


# In[ ]:



