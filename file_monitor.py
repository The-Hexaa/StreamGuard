# file_monitor.py
import time
import subprocess
import os

mail_sent = []
def on_modified():
    output_file = os.path.abspath('static/output_video.mp4')
    
    if os.path.exists(output_file):
        os.remove(output_file)
    
    subprocess.run([
        'ffmpeg', 
        '-i', 'content/record/output.webm', 
        '-c:v', 'libx264', 
        '-crf', '23', 
        '-preset', 'medium', 
        '-c:a', 'aac', 
        '-b:a', '192k', 
        'static/output_video.mp4'
    ])