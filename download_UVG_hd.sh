VIDEOS=("Beauty" "Bosphorus" "HoneyBee" "Jockey" "ReadySetGo" "ShakeNDry" "YachtRide")
for ((i=0; i<${#VIDEOS[@]}; i++)) do
    VIDEO=${VIDEOS[i]}
    DATA_PTH="data/UVG/${VIDEO}_1080p"
    FILE="${VIDEO}_1920x1080_120fps_420_8bit_YUV"
    echo "Downloading UVG ${VIDEO} ($((i+1)) / 7)"
    mkdir -p $DATA_PTH
    curl https://ultravideo.fi/video/${FILE}_RAW.7z --output $DATA_PTH/out.7z
    uv run py7zr x $DATA_PTH/out.7z $DATA_PTH
    if [ $VIDEO = "ReadySetGo" ]; then
        FILE="ReadySteadyGo_1920x1080_120fps_420_8bit_YUV" # the website has 2 different names
    fi
    rm $DATA_PTH/*.txt $DATA_PTH/out.7z # remove copyright file and zip file
    ffmpeg -s 1920x1080 -pix_fmt yuv420p -framerate 50 -i ${DATA_PTH}/${FILE}.yuv $DATA_PTH/frame_%05d.png
    rm ${DATA_PTH}/${FILE}.yuv
done
