import cv2



def plot(image, landmarks, output):
    for i in range(0, len(landmarks), 2):
        cv2.circle(image, (int(landmarks[i]), int(landmarks[i+1])), 2, (128, 0, 255), 2)
    cv2.imwrite(output, image)
