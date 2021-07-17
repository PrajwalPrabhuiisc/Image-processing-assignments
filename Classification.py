import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Classification.avi', fourcc, 30.0, (640, 480))
while True:
    _, frame = cap.read()
    framec = cv2.resize(frame, (640, 480))
    imgBlur = cv2.GaussianBlur(framec, (9, 9), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(imgGray, 50, 80)
    kernel = np.ones((5, 5))
    imgDilate = cv2.dilate(img_canny, kernel, 2)
    contour, hierarchy = cv2.findContours(imgDilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contour:
        area = cv2.contourArea(c)
        if area > 2500:
            cv2.drawContours(frame, c, -1, (0, 0, 255), 2)
            peri = cv2.arcLength(c, True)
            app = cv2.approxPolyDP(c, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(app)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if 3500 < area < 3800:
                cv2.putText(frame, '1 Rupee Found', (55, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if 4250 < area < 5000:
                cv2.putText(frame, '2 Rupee Found', (55, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if 3900 < area < 4200:
                cv2.putText(frame, '5 Rupee Found', (55, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Coins Found", frame)
    out.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
