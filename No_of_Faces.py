#!/usr/bin/env python
# coding: utf-8

# In[11]:


import cv2
import numpy as np
import dlib
import time

def main():
    # Initialize the video capture object
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()

    # Create a named window for better control
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

    # Variables to calculate FPS
    prev_frame_time = 0
    new_frame_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for i, face in enumerate(faces):
                x, y = face.left(), face.top()
                x1, y1 = face.right(), face.bottom()
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
                cv2.putText(frame, f'Face {i+1}', (x-10, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(f"Face {i+1}: {face}")

            # Calculate FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # Display FPS on the frame
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('Frame', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# In[ ]:




