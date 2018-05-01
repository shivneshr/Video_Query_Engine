#!/bin/sh
cd ../RGBRelation/
rm ../ui/rgb/$1.mp4
ffmpeg -framerate 2 -start_number 001 -s 352*288 -i video-%03d.png -c:v libx264 -pix_fmt yuv420p $1.mp4
rm *.png
mv $1.mp4 ../ui/rgb/