# file_monitor.py
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import aiosmtplib
from email.message import EmailMessage
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

async def send_mail_to_admin(filename, similarity):
    print("before if mail_sent")
    if filename not in mail_sent:
        mail_sent.append(filename)
        print("Mail sent")

        message = EmailMessage()
        message["From"] = "muhammadumer@thehexaa.com"  # Replace with your email
        message["To"] = "mumertrade8@gmail.com"
        message["Subject"] = "Thief detected"
        message.set_content(f"person you tagged with --> filename -->'{filename}' was found today with similarity {similarity}. Go check it out")


        # print(f" love person you tagged with --> filename -->'{filename}' was found today with similarity {similarity}. Go check it out")
        await aiosmtplib.send(
                message,
                hostname="sandbox.smtp.mailtrap.io",  # Replace with your SMTP server
                port=2525,
                start_tls=True,
                username="c47084a8353db0",  # Replace with your SMTP username
                password="88af059e5fe011"  # Replace with your SMTP password
            )