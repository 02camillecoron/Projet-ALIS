#Import NWP
import os
import pandas as pd
import numpy as np
import transformers
import numpy
import string
import flask
import torch



#IMPORT GODEL
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



#IMPORT API
from typing import Union
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json




#API ET AUTORISATIONS
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



#############################   NWP   ###########################################################""


top_k = 5
list_proposition=[]

from transformers.models.camembert.modeling_camembert import CamembertForMaskedLM
from transformers.models.camembert.tokenization_camembert import CamembertTokenizer

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")
model.eval()

def get_prediction_eos(input_text):
    try:
        input_text += ' <mask>'#mask token for BERT input 
        res = get_all_predictions(input_text, top_clean=int(top_k))
        
        return res
        
    except Exception as error:
        pass



def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'## if <mask> is the last token, append a "." so that models dont predict punctuation.  
        input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])#input ids tensor
        mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0] #here we are getting ids for masked words
    return input_ids, mask_idx


def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'#this are the tokens which are to be ignored and not to be used.
    tokens = []#empty lsit of tokens
    for w in pred_idx:#loop to itrate input index.
        token = ''.join(tokenizer.decode(w).split())#this will decode the input id and save it in token variable.
        if token not in ignore_tokens:#this loop will check if the word belongs to the token which is to be ignored.
            tokens.append(token.replace('##', ''))#this will replace the ## tags if present in the words and will append the words into token list.
    return ' '.join(tokens[:top_clean])#this will return top K words in the tokens list



def get_all_predictions(text_sentence, top_clean=5):
    
    input_ids, mask_idx = encode(tokenizer, text_sentence)#we will convert input text to encode form i.e tensors.
    
    with torch.no_grad():
        predict = model(input_ids)[0]#we will pass all the input ids to bert model.
    predict_word = decode(tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    predict_tokenize = predict_word.split()
    
    print(predict_word)
    return predict_tokenize




def get_prediction_eos(input_text):
    try:
        input_text += ' <mask>'#masked token is added to input text.
        results = get_all_predictions(input_text, top_clean=int(top_k))# here the function will do encoding decoding and produce the results.
        return results
    except Exception as error:
        pass




#############################   GODEL   ###########################################################""

#CODE IA

#TRADUCTION
cog_key = 'c6a7ac88371842b08e73ef72e22d335d'
cog_location = 'francecentral'
print('Ready to use cognitive services in {} using key {}'.format(cog_location, cog_key))

def translate_text(cog_location, cog_key, text, to_lang='fr', from_lang='en'): 
    import requests, uuid, json 
    #Create the URL for the Text Translator service REST request 
    path = 'https://api.cognitive.microsofttranslator.com/translate?api-version=3.0' 
    params = '&from={}&to={}'.format(from_lang, to_lang) 
    constructed_url = path + params 
    # Prepare the request headers with Cognitive Services resource key and region 
    headers = { 'Ocp-Apim-Subscription-Key': cog_key,
                'Ocp-Apim-Subscription-Region':cog_location,
                'Content-type': 'application/json',
                'X-ClientTraceId': str(uuid.uuid4())
                } 
    # Add the text to be translated to the body 
    body = [{ 
        'text': text 
            }] 
    # Get the translation 
    request = requests.post(constructed_url, headers=headers, json=body) 
    response = request.json() 
    return response[0]["translations"][0]["text"]

#GÉNÉRATION DU MODÈLE
tokenizerGodel = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
modelGodel = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

def generate3(instruction, knowledge, dialog):
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizerGodel(f"{query}", return_tensors="pt").input_ids
    outputs = modelGodel.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizerGodel.decode(outputs[0], skip_special_tokens=True)
    return output

dialog3 = [] # initialisation de la liste vide qui compose l'historique

#FONCTION GODEL PERMETTANT DE GÉNÉRER UNE LISTE DE 3 PROPOSITIONS DE RÉPONSES 
def godel(question): 
#Définition des variables à utiliser 
    instruction = f'Instruction: given a dialog context, you need to response empathically.'
    knowledge = ''
    liste_rep = []
    liste_rep_fr = []
    translation = translate_text(cog_location, cog_key, question, to_lang='en', from_lang='fr')
    dialog3.append(translation)
    #Génération de 3 réponses
    print("\nRéponses possibles : ")
    for i in range(3):
        response_GODEL3 = generate3(instruction, knowledge, dialog3)
        #print(f"Answer {i+1}: {response_GODEL3}")
        liste_rep.append(response_GODEL3)
    #return(liste_rep)
        #Affichage de la traduction des 3 réponses
        translation2 = translate_text(cog_location, cog_key, response_GODEL3, to_lang='fr', from_lang='en')
        print(f"Réponse anglais {i+1}: {response_GODEL3}")        
        print(f"Réponse {i+1}: {translation2}") 
        liste_rep_fr.append(translation2)
        print("\n")
    return(liste_rep_fr)




import os
from datetime import datetime




def sauvegarde(dialog3, id2):
    chemin1 = "test-chemin"
    chemin = os.path.join(chemin1, id2)
    now = datetime.now()
    print("CHEMIN : ", chemin)
    filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
    #dialog3_str = ' '.join(dialog3)
    dialog3_str = dialog3
    try:
        if not os.path.exists(chemin):
            os.makedirs(chemin)
        with open(os.path.join(chemin, filename), 'w') as historique:
            historique.write(dialog3_str)
    except OSError:
        print(f"Error: Failed to create directory or write to file {os.path.join(chemin, filename)}")





def creation_doss(nouveau_dossier):
    dossier_parent = "test-chemin"
    try:
        chemin_complet = os.path.join(dossier_parent, nouveau_dossier)
        if not os.path.exists(chemin_complet):
            os.makedirs(chemin_complet)
            print(f"Le dossier '{chemin_complet}' a été créé avec succès !")
        else:
            print(f"Le dossier '{chemin_complet}' existe déjà.")
    except OSError:
        print(f"Error: Failed to create directory {chemin_complet}")


#APPEL DE L'API
class Question (BaseModel):
    question : str

@app.post('/prediction')
async def godel_answers(question: Question):
    question = question.question
    answers_G = godel(str(question))
    return answers_G

@app.post('/nwp')
async def nwp(question: Question):
    question = question.question
    print("questions : ", question)
    answers_N = get_prediction_eos(str(question))
    print("ANSWERS : ", answers_N)
    #answers = dialoGPT(str(question))
    return answers_N


@app.post('/save')
async def save(request: Request, question: Question):
    question = question.question
    id2 = request.query_params.get('id2') # récupérer l'identifiant depuis l'URL
    print("HISTORIQUE : ", question)
    historique = sauvegarde(str(question), id2) # utiliser le nouvel identifiant pour sauvegarder le fichier


@app.post('/creation')
async def create(question: Question):
    question = question.question
    print("CREATION Nom dossier : ", question)
    doss = creation_doss(str(question))





















# #Import NWP
# import os
# import pandas as pd
# import numpy as np
# import transformers
# import numpy
# import string
# import flask
# import torch





# #IMPORT API
# from typing import Union
# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# import json






# #API ET AUTORISATIONS
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )



# #############################   NWP   ###########################################################""


# top_k = 5
# list_proposition=[]

# from transformers.models.camembert.modeling_camembert import CamembertForMaskedLM
# from transformers.models.camembert.tokenization_camembert import CamembertTokenizer

# tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
# model = CamembertForMaskedLM.from_pretrained("camembert-base")
# model.eval()

# def get_prediction_eos(input_text):
#     try:
#         input_text += ' <mask>'#mask token for BERT input 
#         res = get_all_predictions(input_text, top_clean=int(top_k))
        
#         return res
        
#     except Exception as error:
#         pass



# def encode(tokenizer, text_sentence, add_special_tokens=True):
#     text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
#     if tokenizer.mask_token == text_sentence.split()[-1]:
#         text_sentence += ' .'## if <mask> is the last token, append a "." so that models dont predict punctuation.  
#         input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])#input ids tensor
#         mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0] #here we are getting ids for masked words
#     return input_ids, mask_idx


# def decode(tokenizer, pred_idx, top_clean):
#     ignore_tokens = string.punctuation + '[PAD]'#this are the tokens which are to be ignored and not to be used.
#     tokens = []#empty lsit of tokens
#     for w in pred_idx:#loop to itrate input index.
#         token = ''.join(tokenizer.decode(w).split())#this will decode the input id and save it in token variable.
#         if token not in ignore_tokens:#this loop will check if the word belongs to the token which is to be ignored.
#             tokens.append(token.replace('##', ''))#this will replace the ## tags if present in the words and will append the words into token list.
#     return ' '.join(tokens[:top_clean])#this will return top K words in the tokens list



# def get_all_predictions(text_sentence, top_clean=5):
#     #test_list = []
#     input_ids, mask_idx = encode(tokenizer, text_sentence)#we will convert input text to encode form i.e tensors.
#     # since we will not do backward propogation we will not use gradient calculations.
#     with torch.no_grad():#https://pytorch.org/docs/stable/generated/torch.no_grad.html 
#         predict = model(input_ids)[0]#we will pass all the input ids to bert model.
#     predict_word = decode(tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)#encoded words will now get converted back to words from 0th index to index of mask_ids.and this the magic where the contex will be captured as this is the pretrained BERT model and we will get sensible or meaninfull word at masked_ids place.
#     predict_tokenize = predict_word.split()
#     #test_list.append(predict_tokenize)
#     #print("TEST LISTE : ", test_list)
#     print(predict_word)
#     return predict_tokenize
# # step 5) here we will use above function to encode the input text.



# def get_prediction_eos(input_text):
#     try:
#         input_text += ' <mask>'#masked token is added to input text.
#         results = get_all_predictions(input_text, top_clean=int(top_k))# here the function will do encoding decoding and produce the results.
#         return results
#     except Exception as error:
#         pass




# #APPEL DE L'API
# class Question (BaseModel):
#     question : str

# @app.post('/nwp')
# async def nwp(question: Question):
#     question = question.question
#     print("questions : ", question)
#     answers_N = get_prediction_eos(str(question))
#     print("ANSWERS : ", answers_N)
#     #answers = dialoGPT(str(question))
#     return answers_N