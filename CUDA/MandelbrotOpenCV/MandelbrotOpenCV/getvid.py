from subprocess import call
import os

os.chdir("output")
call([
  "ffmpeg",
  "-framerate", "60",
  "-i", "%d.png",
  "-crf", "14",
  "../zoom.mp4"
  ])