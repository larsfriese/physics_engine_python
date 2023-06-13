# https://www.youtube.com/watch?v=K1Ne_pGy2e8
import cv2, numpy, pyautogui, keyboard

filename = "record"
screen_size = (1920, 1080)
codec = cv2.VideoWriter_fourcc(*'mp4v')
vid = cv2.VideoWriter(filename + ".mp4", codec, 60.0, (screen_size))

while True:
    if keyboard.is_pressed('x'):
        break
    img = pyautogui.screenshot()
    frame = numpy.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    vid.write(frame)

cv2.destroyAllWindows()
vid.release()