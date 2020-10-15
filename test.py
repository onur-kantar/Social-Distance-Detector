# import required libraries
from vidgear.gears import ScreenGear
import cv2

# define dimensions of screen w.r.t to given monitor to be captured
options = {'top': 40, 'left': 0, 'width': 100, 'height': 100}

# open video stream with defined parameters
stream = ScreenGear(monitor=1, logging=True, **options).start()

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break


    # {do something with the frame here}


    # Show output window
    cv2.imshow("Output Frame", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()