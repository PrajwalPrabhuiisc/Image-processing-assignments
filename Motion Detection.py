import cv2
import winsound
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()
    dif = cv2.absdiff(frame2, frame1)
    gray = cv2.cvtColor(dif, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray, 9)
    ret, thres = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thres, None, iterations=3)
    contor, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame1, contor, -1, (0,0,255), 3)
    for c in contor:
        if cv2.contourArea(c) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # winsound.Beep(1000, 200)
        text = "Movement Detected"
        cv2.putText(frame1, "Room Status: {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if cv2.waitKey(10) == ord('q'):
        break
    cv2.imshow('Video Capture', frame1)
