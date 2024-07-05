import keyboard
import threading
import cv2
import mediapipe as mp
import numpy as np
import time
import websocket
import json
import speech_recognition as sr


"""
Smart HACCP 솔루션과 음성 및 모션 인식 기술 연동 코드 (websocket 통신)
"""

"""
음성 인식 명령어 정리
1. 홈 화면: "메인 화면"
2. 새로고침: "새로고침"
3. 뒤로: "페이지 뒤로"
4. 앞으로: "페이지 앞으로"
5. 품목명: "품목명"
6. 품목 선택 화면에서 왼쪽으로 넘김: "왼쪽으로 넘김"
7. 품목 선택 화면에서 오른쪽으로 넘김: "오른쪽으로 넘김"
8. 품목 선택 다중 보기에서 선택 시: "선택 #번"
9. 생산량 선택: "#kg"
10. 음성 인식 종료: "음성 인식 종료"
"""

# Flags to control motion and voice recognition
motion_flag = False
voice_flag = False

def run_voice():
    """Function to run voice recognition"""
    global voice_flag
    voice_flag = True
    server_url = "ws://localhost:8080/recognize/server/websocket"
    ws = websocket.WebSocket()

    def connect_ws():
        # Connect to WebSocket server
        ws.connect(server_url)
        send_message = {
                "code": "10"
            }
        print("Sending data:", send_message)
        ws.send(json.dumps(send_message))

    def send_message(command):
        """Send voice command to WebSocket server"""
        command = command.replace(" ", "")
        if command[-2:] == '화면':
            data = {
                "code": "00"
            }
        elif command[-2:] == '고침':
            data = {
                "code": "01"
            }
        elif command[-2:] == '뒤로':
            data = {
                "code": "02"
            }
        elif command[-3:] == '앞으로':
            data = {
                "code": "03"
            }
        elif command[:2] == '왼쪽':
            data = {
                "code": "21",
                "message": "prev"
            }
        elif command[:3] == '오른쪽':
            data = {
                "code": "21",
                "message": "next"
            }
        elif command[-1:] == '번':
            data = {
                "code": "12",
                "message": command[-2:-1]
            }
        else:
            data = {
                "code": "11",
                "message": command
            }
        ws.send(json.dumps(data))
        print("Sending data:", data)

    def my_voice():
        """Capture and recognize voice command"""
        global voice_flag
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("[ Starting voice recognition ]")
            audio = r.listen(source)

        if voice_flag == True:
            try:
                text = r.recognize_google(audio, language='ko-KR')
                print(f"Recognized voice: {text}")
                return text
            except sr.UnknownValueError:
                print("Voice recognition failed!")
                return None
            except sr.RequestError as e:
                print(f"Voice recognition service error: {e}")
                return None
        else:
            return "stop"

    def continuous_recognition():
        """Continuously recognize voice commands"""
        global voice_flag
        while voice_flag:
            command = my_voice()
            if command:
                if command[-2:] == "종료":
                    print("[ Stopping voice recognition ]")
                    voice_flag = False
                    break
                else:
                    if command == "stop":
                        break
                    else:
                        send_message(command)
            time.sleep(1)

    connect_ws()
    continuous_recognition()
    ws.close()

def run_motion():
    """Function to run motion recognition using webcam and MediaPipe"""
    print("[ Starting motion recognition ]") 
    global motion_flag
    motion_flag = True

    # Maximum number of hands to detect
    max_num_hands = 1

    # Gesture mapping
    gesture = {
        0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
        6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'selected',
        11: 'left', 12: 'right', 13: 'home', 14: 'back', 15: 'front', 16: 'prev', 17: 'next'
    }

    # Initialize MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # Load gesture recognition model
    file = np.genfromtxt('data/motion_dataset.csv', delimiter=',')
    angle = file[:, :-1].astype(np.float32)
    label = file[:, -1].astype(np.float32)
    knn = cv2.ml.KNearest_create()
    knn.train(angle, cv2.ml.ROW_SAMPLE, label)

    # Connect to WebSocket server
    server_url = "ws://localhost:8080/recognize/server/websocket"
    ws = websocket.WebSocket()

    def connect_ws():
        ws.connect(server_url)
        send_message = {
                "code": "20"
            }
        print("Sending data:", send_message)
        ws.send(json.dumps(send_message))

    connect_ws()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def send_gesture(gesture_idx):
        """Send gesture data to WebSocket server"""
        if gesture_idx in [10, 11, 12, 16, 17]:
            send_message = {
                "code": "21",
                "message": gesture[gesture_idx]
            }
        elif gesture_idx == 13:
            send_message = {
                "code": "00"
            }
        elif gesture_idx == 14:
            send_message = {
                "code": "02"
            }
        elif gesture_idx == 15:
            send_message = {
                "code": "03"
            }
        else:
            send_message = {
                "code": "21",
                "message": gesture_idx
            }
        print("Sending data:", send_message)
        ws.send(json.dumps(send_message))

    def compute_angles(res):
        """Compute angles between hand joints"""
        joint = np.zeros((21, 3))
        for j, lm in enumerate(res.landmark):
            joint[j] = [lm.x, lm.y, lm.z]

        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :] 
        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  
        v = v2 - v1  # [20,3]
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        angle = np.arccos(np.einsum('nt,nt->n',
                                     v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                     v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]
        angle = np.degrees(angle)
        return angle

    def infer_gesture(angle_data):
        """Infer gesture from angle data using KNN"""
        data = np.array([angle_data], dtype=np.float32)
        ret, results, neighbours, dist = knn.findNearest(data, 3)
        idx = int(results[0][0])
        return idx

    start_time = time.time()

    while cap.isOpened() and motion_flag:
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                angle_data = compute_angles(res)
                gesture_idx = infer_gesture(angle_data)

                elapsed_time = time.time() - start_time
                if elapsed_time > 4: # wait for 4 Seconds
                    send_gesture(gesture_idx)
                    start_time = time.time()

                cv2.putText(img, text=gesture[gesture_idx].upper(),
                            org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=3)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
    
        cv2.imshow('Hand Tracking', img)
        if cv2.waitKey(1) == ord('m') or keyboard.is_pressed('m'):
            print("[ Stopping motion recognition ]")
            motion_flag = False
            break

    cap.release()
    cv2.destroyAllWindows()
    ws.close()

# 발판 중 왼쪽을 'v', 중간을 'esc', 오른쪽을 'm'으로 키 값 설정
def on_press(event):
    """Event handler for keyboard press"""
    global motion_flag, voice_flag

    if event.name == 'm' and not voice_flag:
        if not motion_flag:
            threading.Thread(target=run_motion).start()

    elif event.name == 'v' and not motion_flag:
        if not voice_flag:
            threading.Thread(target=run_voice).start()
        else:
            print("[ Stopping voice recognition ]")
            voice_flag = False

# Set up keyboard event listener
keyboard.on_press(on_press)
keyboard.wait('esc')
print("[ HACCP connection terminated! ]")
