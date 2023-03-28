import gradio as gr
import openai
import os
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

#accessing openapi Key.
openai.api_key = os.getenv("OPENAI_API_KEY")

audio_messages = [{"role": "system", "content": 'You are an AI assistant expert. Respond to all input in precise, crisp and easy to understand language.'}]
text_messages = [{"role": "system", "content": 'You are an AI assistant expert. Respond to all input in precise, crisp and easy to understand language.'}]
global user_text_input, text_output, user_audio_input, audio_output

"""
It seems like the gr.Audio source is not generating a WAV file, which is required for the openai.Audio.transcribe() method to work. 
To convert the audio file to WAV format, i have used a library like Pydub.
"""

def audio_transcribe(audio):
    global audio_messages
    audio_message = audio_messages

    #audio processing to whisper API.
    audio_file = AudioSegment.from_file(audio)
    audio_file.export("temp.wav", format="wav")
    final_audio_file = open("temp.wav", "rb")
    transcript = openai.Audio.transcribe("whisper-1", final_audio_file)
    os.remove("temp.wav")

    #transcripted input to chatGPT API for chatCompletion 
    audio_message.append({"role": "user", "content": transcript["text"]}) # type: ignore
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=audio_message)
    system_message = response["choices"][0]["message"] # type: ignore
    audio_message.append(system_message)

    chat_transcript = ""
    for message in audio_message:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    return chat_transcript

def text_transcribe(name):
    global text_messages
    text_message = text_messages
    user_text_input.update("")
    #transcripted input to chatGPT API
    text_message.append({"role": "user", "content": name}) # type: ignore
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=text_message)
    system_message = response["choices"][0]["message"] # type: ignore
    text_message.append(system_message)
    
    chat_transcript = ""
    for message in text_message:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"
    return chat_transcript

# def text_clear():
#     user_text_input.update("")
#     text_output.update("")
# def clear_textbox(name, name2):
#     name.update("")
#     name2.update("")
#     # reload the Gradio app
#     print("Reloading app...")
#     js = "window.location.reload();"
#     gr.Interface.fn(js)() # type: ignore

# def audio_clear():
#     user_audio_input.value = None
#     audio_output.value = ""

title = """<h1 align="center">Your chatGPT AI Assistant at your Service!! 😎 </h1>"""
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(title)
    with gr.Tab("Audio Input"):
        with gr.Row():
            user_audio_input = (gr.Audio(source="microphone", type="filepath", label="Speak Here"))
            audio_input = user_audio_input
            audio_output = gr.Textbox(label="AI Response", lines=20, placeholder="AI Response will be displayed here...")
        with gr.Row():
            audio_submit_button = gr.Button("Submit")
            #audio_reset_button = gr.Button("Reset")
    with gr.Tab("Text Input"):
        with gr.Row():
            user_text_input = (gr.Textbox(label="Type Here", lines=20, placeholder="Type your message here..."))
            text_input = user_text_input
            text_output = gr.Textbox(label="AI Response", lines=20, placeholder="AI Response will be displayed here...")
        with gr.Row():
            text_submit_button = gr.Button("Submit")
            #text_reset_button = gr.Button("Reset")
    audio_submit_button.click(fn=audio_transcribe, inputs=audio_input, outputs=audio_output)
    # audio_reset_button.click(fn=audio_clear, inputs=user_audio_input, outputs=audio_output)
    text_submit_button.click(fn=text_transcribe, inputs=text_input, outputs=text_output)
    #text_reset_button.click(fn=text_clear)
    #text_reset_button.click(fn=clear_textbox, inputs=[text_input, text_output])

demo.launch(share=True)
