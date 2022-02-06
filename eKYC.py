from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

import cv2 as cv
import numpy as np

import face_recognition
import os

import shutil

path = "dataset"
images = []
className = []
img_id = 0
myList = os.listdir(path)
for cl in myList:
    curImg = cv.imread(f'{path}/{cl}')
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
encodeListKnown = findEncodings(images)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index_get(request: Request):
    return templates.TemplateResponse("ai-web.html", {"request": request})

@app.websocket("/websocket")
async def websocket_endpoint(websocket: WebSocket):
    path = "dataset"
    images = []
    className = []
    img_id = 0
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv.imread(f'{path}/{cl}')
        images.append(curImg)
        className.append(os.path.splitext(cl)[0])
    def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList
    encodeListKnown = findEncodings(images)

    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            image = cv.imdecode(nparr, -1)
            imgS = cv.resize(image,(0,0),None,0.25,0.25)
            imgS = cv.cvtColor(imgS, cv.COLOR_RGB2BGR)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
            for encodeFace, faceLoc in zip(encodeCurFrame,facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)
                if len(encodeCurFrame) == 2:
                    matches1 = face_recognition.compare_faces(encodeListKnown, encodeCurFrame[0])
                    faceDis1 = face_recognition.face_distance(encodeListKnown, encodeCurFrame[0])
                    matchIndex1 = np.argmin(faceDis1)
                    peple1 = className[matchIndex1].upper()#dol

                    matches2 = face_recognition.compare_faces(encodeListKnown, encodeCurFrame[1])
                    faceDis2 = face_recognition.face_distance(encodeListKnown, encodeCurFrame[1])
                    matchIndex2 = np.argmin(faceDis2)
                    peple2 = className[matchIndex2].upper()#dol.card

                    peple1.split(".")
                    peple2.split(".")
                    if peple1[0] == peple2[0]:
                        await websocket.send_json({
                            'compare' : 'true'
                        })
                    else:
                        await websocket.send_json({
                            'compare' : 'false'
                        })
                if matches[matchIndex]:
                    name = className[matchIndex].upper()
                    name = name.split(".")
                    await websocket.send_json({
                        'name' : name[0]
                    })
                else:
                    await websocket.send_json({
                        'name' : 'unknow'
                    })
        except WebSocketDisconnect:
            await websocket.close()
            print('Closed connection')
            break

@app.get("/upload", response_class=HTMLResponse)
async def get_upload(request: Request):
    return templates.TemplateResponse("upload-page.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def post_upload(request : Request,profile : UploadFile = File(...), card : UploadFile = File(...)):
    with open(f'dataset/{profile.filename}', 'wb') as buffer:
        shutil.copyfileobj(profile.file, buffer)
    with open(f'dataset/{card.filename}', 'wb') as buffer:
        shutil.copyfileobj(card.file, buffer)
    return templates.TemplateResponse("ai-web.html", {"request": request})
