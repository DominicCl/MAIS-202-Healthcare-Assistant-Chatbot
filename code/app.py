from flask import Flask, render_template
import torch
from chatbot import chatbot

app= Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')