Put your own dashcam clip here.

  data/dashcam.mp4   (preferred default name)

The visualizer picks up data/dashcam.mp4 automatically when --source is omitted.
If no .mp4 is found, it falls back to webcam index 0.

You can also pass any file explicitly:

  python main.py --source path/to/your_clip.mp4 --view split

NO VIDEO FILES ARE SHIPPED IN THE REPO. Recommended source clips:
  * Any dashcam clip you own.
  * Free/Creative Commons stock at pexels.com/videos, pixabay.com/videos, mixkit.co.
  * Trim a YouTube clip you have rights to with: yt-dlp <URL> --download-sections "*10-20"
    (For PERSONAL testing only — don't redistribute downloaded copyrighted material.)

After dropping a clip, optionally tune lane boundaries in-app: launch, press
[, ], `,`, `.` to adjust, then S to save data/lane_calibration.json so the
next run loads your calibration automatically.
