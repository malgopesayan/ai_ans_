from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import threading
import google.generativeai as genai
from openai import OpenAI
import groq
import re
import os
import base64
import time
from PIL import Image
import io
from threading import Lock

app = Flask(__name__)
run_with_ngrok(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'supersecretkey'

# API configuration
API_KEYS = {
    "GEMINI_ANSWER": "AIzaSyCdNncadq3sU6CGig-PCNH0nCgFAEW6WGM",
    "GEMINI_TEXT": "AIzaSyC-h_vxfoYSO8gc7-XWwekjYNTAya0-Ocw",
    "NVIDIA": "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",
    "GROQ": "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",
    "GITHUB": "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"
}

# Global variables for processing state
processing_lock = Lock()
current_results = {}
processing = False
extracted_text = ""
final_prompt = ""

# Initialize clients
groq_client = groq.Client(api_key=API_KEYS["GROQ"])
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEYS["NVIDIA"]
)
azure_client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=API_KEYS["GITHUB"]
)

@app.route('/')
def splash():
    return render_template('splash.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    global processing, extracted_text, final_prompt
    processing = True
    current_results.clear()
    
    # Save image
    image_data = request.json['image'].split(',')[1]
    img = Image.open(io.BytesIO(base64.b64decode(image_data)))
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'current.jpg')
    img.save(img_path)
    
    # Start processing threads
    threading.Thread(target=process_image, args=(img_path,)).start()
    
    return jsonify({'status': 'processing'})

@app.route('/results')
def get_results():
    return jsonify(current_results)

def process_image(img_path):
    global current_results, extracted_text, final_prompt, processing
    
    try:
        # Gemini Answer
        current_results['Gemini'] = gemini_answer(img_path)
        
        # Text extraction
        extracted_text = gemini_text_extract(img_path)
        final_prompt = f"""please Give only the correct answer to this question, no explanation, like 'the correct answer is: a)19'
        Question: {extracted_text}"""
        
        # Process other models
        models = [
            ('NVIDIA', process_nvidia),
            ('Deepseek', process_deepseek),
            ('Llama', process_llama),
            ('GPT-4o', process_gpt4o)
        ]
        
        threads = []
        for name, func in models:
            thread = threading.Thread(target=run_model, args=(name, func))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
    except Exception as e:
        current_results['error'] = str(e)
    finally:
        processing = False

def run_model(name, func):
    try:
        result = func()
        current_results[name] = result
    except Exception as e:
        current_results[name] = f"Error: {str(e)}"

def gemini_answer(img_path):
    genai.configure(api_key=API_KEYS["GEMINI_ANSWER"])
    img = Image.open(img_path)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content([
        "please Give only the correct answer to this question, no explanation, like 'the correct answer is: a)19'",
        img
    ])
    return response.text.strip()

def gemini_text_extract(img_path):
    genai.configure(api_key=API_KEYS["GEMINI_TEXT"])
    img = Image.open(img_path)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content([
        "Extract all text exactly as it appears in the image",
        img
    ])
    return response.text

def process_nvidia():
    response = nvidia_client.chat.completions.create(
        model="nvidia/nemotron-4-340b-instruct",
        messages=[{"role": "user", "content": final_prompt}]
    )
    return clean_response(response.choices[0].message.content)

def process_deepseek():
    response = groq_client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[{"role": "user", "content": final_prompt}]
    )
    return clean_response(response.choices[0].message.content)

def process_llama():
    response = groq_client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[{"role": "user", "content": final_prompt}]
    )
    return clean_response(response.choices[0].message.content)

def process_gpt4o():
    response = azure_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": final_prompt}]
    )
    return clean_response(response.choices[0].message.content)

def clean_response(text):
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"Correct Answer:\s*(.*)", cleaned, re.IGNORECASE)
    return match.group(0).strip() if match else cleaned.strip()

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
