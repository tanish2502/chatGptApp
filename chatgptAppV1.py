import gradio as gr
import openai
import config
import os, subprocess
from pydub import AudioSegment

#accessing openapi Key.
openai.api_key = config.OPENAI_API_KEY

messages = [{"role": "system", "content": 'You are an AI assistant expert. Respond to all input in precise, crisp and in short as much as possible.'}]

"""
It seems like the gr.Audio source is not generating a WAV file, which is required for the openai.Audio.transcribe() method to work. 
To convert the audio file to WAV format, you can use a library like Pydub.
"""
def transcribe(audio):
    global messages

    #audio processing to whisper API.
    audio_file = AudioSegment.from_file(audio)
    audio_file.export("temp.wav", format="wav")
    final_audio_file = open("temp.wav", "rb")
    transcript = openai.Audio.transcribe("whisper-1", final_audio_file)
    print(transcript)
    os.remove("temp.wav")

    #transcripted input to chatGPT API
    messages.append({"role": "user", "content": transcript["text"]}) # type: ignore
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    system_message = response["choices"][0]["message"] # type: ignore
    messages.append(system_message)

    #subprocess.call(["say", system_message['content']])
    
    chat_transcript = ""
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    return chat_transcript
    #return transcript["text"] # type: ignore

audio_input = (gr.Audio(source="microphone", type="filepath", label="Speak Here").style())
text_input = (gr.Textbox(label="Type Here", lines=2, placeholder="Type your message here..."))

ui = gr.Interface(fn=transcribe, inputs=[audio_input,text_input], outputs="text")



ui.launch()
