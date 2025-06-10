mkdir -p data/DAVIS
curl https://cgl.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip --output data/DAVIS/DAVIS.zip
unzip -qq data/DAVIS/DAVIS.zip -d data/DAVIS/
rm data/DAVIS/DAVIS.zip
VIDEOS=("blackswan"  "bmx-trees"  "boat"  "breakdance"  "camel"  "car-roundabout"  "car-shadow"  "cows"  "dance-twirl"  "dog")

for ((i=0; i<${#VIDEOS[@]}; i++)) do
    VIDEO=${VIDEOS[i]}
    mv data/DAVIS/DAVIS/JPEGImages/1080p/${VIDEO} data/DAVIS/${VIDEO}
done
rm -r data/DAVIS/DAVIS
