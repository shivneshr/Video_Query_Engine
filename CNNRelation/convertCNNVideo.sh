cd ../CNNRelation/
cd raw
ffmpeg -framerate 2 -start_number 001 -s 352*288 -i video-%03d.jpg -c:v libx264 -pix_fmt yuv420p $1

mv $1 ../../ui/CNN/$1