mkdir -p data/YOUTUBE_8M/MarioKart
yt-dlp -f "bestvideo[height<=720]" -o "data/YOUTUBE_8M/MarioKart.%(ext)s" --no-playlist https://www.youtube.com/watch?v=4yZlK2Ftjho
ffmpeg -i data/YOUTUBE_8M/MarioKart.mp4 -vframes 4000 data/YOUTUBE_8M/MarioKart/f%05d.png
rm data/YOUTUBE_8M/MarioKart.mp4
