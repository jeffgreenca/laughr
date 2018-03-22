@echo off
REM Given a video clip, mute the laughs using laughr.py and create
REM a new copy of the video file with the new audio track.
echo Usage: laughr-helper.bat model.h5 source.avi output.mp4
set modelfile=%1
set infile=%2
set outfile=%3

ffmpeg -i %infile% temp.wav
python laughr.py --model %modelfile% --mute-laughs temp.wav temp-out.wav
ffmpeg -i %infile% -i temp-out.wav -map 0:0 -map 1:0 -c:v copy -c:a aac -b:a 256k -shortest %outfile%

echo .
echo You may want to delete temp.wav and temp-out.wav
