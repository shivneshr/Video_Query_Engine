cd ../CNNRelation/
cd raw
rm *.mp4
ffmpeg -framerate 2 -start_number 001 -s 352*288 -i video-%03d.jpg -c:v libx264 -pix_fmt yuv420p CNNMatch.mp4