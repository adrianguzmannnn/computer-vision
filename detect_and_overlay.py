import cv2
import os
import random

# Load the frontal face classifier.
classifier = r'.\classifiers\haarcascade_frontalface_alt.xml'
if not os.path.isfile(classifier):
    error = 'File `{}` does not exist. Refer to the GitHub page for ' \
            'requirements.'.format(classifier)
    raise IOError(error)
face_cascade = cv2.CascadeClassifier(classifier)

# Begin a video capture. The index of `0` denotes that the default camera
# is selected.
video_capture = cv2.VideoCapture(0)

# Ensure that a camera was found. Otherwise, raise an exception.
if not video_capture:
    raise IOError('A default camera was not identified.')

# Retrieve the camera's specifications.
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
dimensions = (frame_width, frame_height)
fps = int(video_capture.get(5))

# Configure the output of the recording. FourCC is a 4-byte code used to define
# the codec; it is operating-system dependent. More details at `fourcc.org`.
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('output.avi', fourcc, fps, dimensions)

# Load the overlay images. Raise an exception if none are found.
images = [cv2.imread(os.path.join('media', f), -1) for f in os.listdir('media')
          if f.endswith('.png')]
if len(images) == 0:
    raise IOError('No PNG files were found.')

original_masking = [image[:, :, 3] for image in images]

# Generate the inverted masking. `cv2.bitwise_not()` inverts every bit of
# an array.
inverted_masking = [cv2.bitwise_not(mask) for mask in original_masking]

# Unless the user has specified to terminate the recording, continue it
# using this while loop.
while True:

    # Begin reading the video capture.
    ret, frame = video_capture.read()

    # Convert the video feed to gray-scale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use a haar-cascade to detect faces in the feed. The arguments as
    # described as:
    # a. `scaleFactor` specifies how much the image size is reduced at each
    #    image scale.
    # b. `minNeighbors` defines how many neighbors each candidate
    #    rectangle should
    #    have to retain it.
    # c. `minSize` specifies the minimum pixel area to be considered for
    #    detection.
    faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(30, 30)
        )

    # Iterate through each face found.
    for (x, y, w, h) in faces:

        # Randomly select an index depending on the number of PNGs available
        # to overlay.
        index = random.randint(0, len(images)-1)

        # Define the overlay pixel area.
        overlay_width = int(w*2.4)
        overlay_height = overlay_width
        overlay_dimensions = (overlay_width, overlay_height)

        # Re-size the overlay PNG image and maskings to the defined dimensions.
        overlay = cv2.resize(images[index][:, :, 0:3], overlay_dimensions,
                             interpolation=cv2.INTER_AREA)
        mask = cv2.resize(original_masking[index], overlay_dimensions,
                          interpolation=cv2.INTER_AREA)
        inverted_mask = cv2.resize(inverted_masking[index], overlay_dimensions,
                                   interpolation=cv2.INTER_AREA)

        # Define the coordinates for the overlay.
        x1 = int(x-(overlay_width*0.325))
        x2 = x1+overlay_width
        y1 = int(y-(overlay_height*0.35))
        y2 = y1+overlay_height

        # Generate the overlay.
        try:
            # Filter the frame to the region of interest, roi.
            roi = frame[y1:y2, x1:x2]
            inverted_roi = cv2.bitwise_and(roi, roi, mask=inverted_mask)
            overlay_roi = cv2.bitwise_and(overlay, overlay, mask=mask)

            # Join the filtered matrices.
            merge = cv2.add(inverted_roi, overlay_roi)

            # Overwrite the frame and write the output.
            frame[y1:y2, x1:x2] = merge
            output.write(frame)
        except:
            pass

    # Initialize a window and set it as full-screen.
    cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    # Show the initialized window.
    cv2.imshow('Video', frame)

    # Press 'Q' to end the session.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Terminate the video live capture and recording.
output.release()
video_capture.release()
cv2.destroyAllWindows()
