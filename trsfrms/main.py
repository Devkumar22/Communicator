from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import speech_recognition as sr
import pyttsx3
import pyaudio as py
import streamlit as st
from PIL import Image
# import pywhatkit



# def main():
    # with sr.Microphone() as source:
    #     listener.adjust_for_ambient_noise(source,duration=1)
    #     print('listening...')
    #     voice = listener.listen(source)
    #     command = listener.recognize_google(voice)        #using google api to send audio and google will send the text
    #     mic_in = str(command)
uploaded_file = st.file_uploader("Choose a file")
img = Image.open("cool.jpg")
if uploaded_file:
    st.image(uploaded_file)

engine = pyttsx3.init()

listener = sr.Recognizer()
        
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")#.to("cuda")

# Let's chat for 5 lines
for step in range(10):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    #new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')#.to("cuda")


    ########################################
    with sr.Microphone() as source:
        listener.adjust_for_ambient_noise(source,duration=1)
        print('listening...')
        voice = listener.listen(source)
        
        command = listener.recognize_google(voice)        #using google api to send audio and google will send the text
        
        if command:
            print(command)
            mic_in = str(command)
        else:
            st.warning("Please Speak again!")
            continue
        if 'exit' or 'bye' in command:
                break
            

    new_user_input_ids = tokenizer.encode(mic_in + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)#.to("cuda")

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
    engine.say("hey {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
    engine.runAndWait()


# if __name__ == 'main':
#     main()