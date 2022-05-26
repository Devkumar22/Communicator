import speech_recognition as sr
import pyttsx3
import pyaudio as py
import pywhatkit

engine=pyttsx3.init()

listener = sr.Recognizer()

with sr.Microphone() as source:
    listener.adjust_for_ambient_noise(source,duration=1)
    print('listening...')
    voice = listener.listen(source)
    command = listener.recognize_google(voice)        #using google api to send audio and google will send the text
    mic_in = str(command)
    print(mic_in)
    engine.say(mic_in)
    engine.runAndWait()
        