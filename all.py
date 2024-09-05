import yt_dlp

def get_stream_url(video_id):
    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'noplaylist': True,
        'force_generic_extractor': True,
    }
    url = f'https://www.youtube.com/watch?v={video_id}'
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        return info_dict.get('url')

video_id = 'YOUR_VIDEO_ID'
stream_url = get_stream_url(video_id)
print(f"Stream URL: {stream_url}")


import cv2

def main(stream_url):
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow('Live Stream', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(stream_url)

