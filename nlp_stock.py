import spacy
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import json
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

df = pd.read_csv("data/stocks.tsv", sep="\t")
#list of symbols and companies
symbols = df.Symbol.tolist()
companies = df.CompanyName.tolist()

df2 = pd.read_csv("data/indexes.tsv", sep="\t")
#list of stock indexes
indexes = df2.IndexName.tolist()
index_symbols = df2.IndexSymbol.tolist()

df3 = pd.read_csv("data/stock_exchanges.tsv", sep="\t")
exchanges = df3.ISOMIC.tolist() + df3["Google Prefix"].tolist() + df3.Description.tolist()

stops = [] 
nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = []

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
for symbol in symbols:
    stops.append(symbol.lower())
    patterns.append({"label": "STOCK", "pattern": symbol})
    for l in letters:
        patterns.append({"label": "STOCK", "pattern": symbol + f".{l}"})
        #look for any instance where there is a symbol followed by a period and letters
for company in companies:
    stops.append(company.lower())
    if company not in stops:
        patterns.append({"label": "COMPANY", "pattern": company})
for index in indexes:
    stops.append(index.lower())
    patterns.append({"label": "INDEX", "pattern": index})
    words = index.split()
    patterns.append({"label": "INDEX", "pattern": " ".join(words[:2])}) #hard coded to identify S&P 500 as index
    patterns.append({"label": "INDEX", "pattern": " ".join(words[:1])})
for index_symbol in index_symbols:
     stops.append(index_symbol.lower())
     patterns.append({"label": "INDEX", "pattern": index})
for e in exchanges:
    if pd.isna(e): #check if exchange is NaN
        continue
    stops.append(e.lower())
    patterns.append({"label": "STOCK_EXCHANGE", "pattern": e})
ruler.add_patterns(patterns)
#add pattern to entity ruler

url = "https://www.goldmansachs.com/insights/articles/how-us-fiscal-concerns-are-affecting-bonds-currencies-stocks"
page = requests.get(url)
text = bs(page.text, "html.parser")
doc = nlp(text.text)

sents = list(doc.sents)
doc_length = len(sents)

event = []
person = []
company = []
org = []
stock = []
index = []

for ent in doc.ents:
    #if ent == "EVENT"
    if ent.label_ == "PERSON" and ent.text not in person:
        person.append(ent.text)
    if ent.label_ == "COMPANY" and ent.text not in company:
        company.append(ent.text)
    if ent.label_ == "ORG" and ent.text not in org:
        org.append(ent.text)
    if ent.label_ == "STOCK" and ent.text not in stock:
        org.append(ent.text)
    if ent.label_ == "INDEX" and ent.text not in index:
        org.append(ent.text)
        
            
print("relevant people: " + ', '.join(person))
print("relevant companies: " + ', '.join(company))
print("relevant organizations: " + ', '.join(org))
print("relevant stocks: " + ', '.join(stock))
print("relevant indexes: " + ', '.join(index))


model = 'facebook/bart-large-cnn'
transformer = BartForConditionalGeneration.from_pretrained(model)
tokenizer = BartTokenizer.from_pretrained(model)

final_text = ""

#marks text as important
for s in sents:
    if any(s.ents in symbols or indexes or exchanges or ent.label_ == "PERCENTAGE"):
        final_text += (f"[IMPORTANT] {s}")
    else: 
        final_text += s 
#bypass token limit by splitting up the tokens

#1: chunk up the large text into smaller token chunks -> use a list to store the chunks to iterate (easy and simple ds to use)
#2: loop through each chunk and append each iteration into some variable of final_message

#1
tokens = tokenizer.encode(final_text) #create a list of tokens from the text
chunk_tokens = []
for i in range(0, len(tokens), 1024): #1024 is the max amount of tokens the BART model can take
    chunk_tokens.append(tokens[i:i+1024])

#2
final_summary = ""
for chunk in chunk_tokens:
    inputs = {'input_ids': torch.tensor([chunk])}

    summary_ids = transformer.generate(
        inputs['input_ids'], min_length=40, num_beams=4, early_stopping=True
    )

    final_summary += tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("\nSummary: " + final_summary)

#debug
# from spacy import displacy
# doc = nlp(text.text)
# print(displacy.render(doc, style="ent", page="true"))


