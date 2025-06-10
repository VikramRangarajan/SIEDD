mkdir -p data/UVG_Bunny/Bunny_720p # not UVG but it's fine
curl https://raw.githubusercontent.com/scikit-video/scikit-video/master/skvideo/datasets/data/bigbuckbunny.mp4 --output data/UVG_Bunny/bunny.mp4
ffmpeg -i data/UVG_Bunny/bunny.mp4 data/UVG_Bunny/Bunny_720p/f%05d.png
rm data/UVG_Bunny/bunny.mp4
