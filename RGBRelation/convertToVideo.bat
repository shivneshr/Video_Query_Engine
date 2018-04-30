del ..\ui\rgb\*.mp4
ffmpeg -framerate 1 -start_number 001 -s 352*288 -i video-%%03d.png -c:v libx264 -pix_fmt yuv420p rgbMatchedVideo.mp4
ffmpeg -framerate 1 -start_number 001 -s 352*288 -i query-%%03d.png -c:v libx264 -pix_fmt yuv420p rgbMatchedQuery.mp4
del *.png
move *.mp4 ..\ui\rgb\