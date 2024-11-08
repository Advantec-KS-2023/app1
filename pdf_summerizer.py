import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from PyPDF2 import PdfReader
import os, torch, gc, requests, re

MODEL_PATH = "tsmatz/mt5_summarize_japanese"
seq2seq = pipeline("summarization", model = MODEL_PATH, max_length = 300, min_length=100)

def extract_the_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = " "
    for page in reader.pages:
        cs = page.extract_text()
        cs2 = cs.replace('\n', ' ')
        text += cs2
    return text.strip()

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



st.title('PDF Summarization（PDF要約化）')

upload_file = st.file_uploader('Choose a PDF file', type='pdf')

if upload_file is not None:
    pdf_content = upload_file.getvalue()
    pdf_path = 'temp.pdf'
    with open(pdf_path, "wb") as f:
        f.write(pdf_content)

    st.subheader('Extract Text: ')
    extractes_text = extract_the_text_from_pdf(pdf_path)
    #extractes_text = extract_the_text_from_url(url_name)
    st.write(extractes_text)

    st.subheader('Summary:')
    summary = seq2seq(extractes_text)[0]
    st.write(summary.get('summary_text'))

    os.remove(pdf_path)
    gc.collect()


st.subheader('How does it works?')
code_to_display = '''
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from PyPDF2 import PdfReader
import os, torch, gc, requests, re

MODEL_PATH = "tsmatz/mt5_summarize_japanese"
seq2seq = pipeline("summarization", model = MODEL_PATH, min_length=500)

def extract_the_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = " "
    for page in reader.pages:
        cs = page.extract_text()
        cs2 = cs.replace('\n', ' ')
        text += cs2
    return text.strip()

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



st.title('PDF Summarization（PDF要約化）')

upload_file = st.file_uploader('Choose a PDF file', type='pdf')

if upload_file is not None:
    pdf_content = upload_file.getvalue()
    pdf_path = 'temp.pdf'
    with open(pdf_path, "wb") as f:
        f.write(pdf_content)

    st.subheader('Extract Text: ')
    extractes_text = extract_the_text_from_pdf(pdf_path)
    #extractes_text = extract_the_text_from_url(url_name)
    st.write(extractes_text)

    st.subheader('Summary:')
    summary = seq2seq(extractes_text)[0]
    st.write(summary.get('summary_text'))

    os.remove(pdf_path)
    gc.collect()
'''
#st.code(code_to_display)



