import torch
import torch.nn.functional as F
import cv2
import numpy as np
from GradCam import GradCAM
from model import SmallCNN
classes = torch.load('./classes.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = SmallCNN(len(classes))
model.load_state_dict(torch.load('./best_model_acc.pth', map_location='cpu'))
model.eval()

gradcam = GradCAM(model)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture('../test/Can you tell when someone is lying to you_(360P).mp4')
target_fps = 15
delay = int(1000 / target_fps)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    org = frame.copy()
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_full, 1.3,5)

    for (x, y, w, h) in faces:
        face = gray_full[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48,48))
        
        input_tensor = torch.tensor(face_resized).unsqueeze(0).unsqueeze(0).float().to(device)

        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1) 
        pred_class = probabilities.argmax(1).item()
        
        conf = probabilities[0, pred_class].item() * 100

        cv2.putText(frame, f"{classes[pred_class]} {conf:.1f}%",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cam = gradcam.generate(input_tensor)
        
        heatmap = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (w,h))
        overlay = cv2.addWeighted(frame[y:y+h, x:x+w], 0.5, heatmap, 0.5, 0)
        frame[y:y+h, x:x+w] = overlay
        
    cv2.imshow("Cascade + Classification", frame)
    cv2.imshow("Grad-CAM Visualization", org)

    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
