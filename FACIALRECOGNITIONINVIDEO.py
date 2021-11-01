"""
Created on Sun May  2 14:28:16 2021
@author: DEELMU TECHNOLOGIES
"""
import face_recognition
import os
import cv2
knownfacesdir = "knownfaces"
tolerance = 0.5
framethickness = 3
fontthickness = 2
MODEL = "hog"
video = cv2.VideoCapture(0)
print("loading known faces")
knownfaces = []
knownnames = []
for name in os.listdir(knownfacesdir):
	for filename in os.listdir(f"{knownfacesdir}/{name}"):
		image = face_recognition.load_image_file(f"{knownfacesdir}/{name}/{filename}")
		encoding = face_recognition.face_encodings(image)[0]
		knownfaces.append(encoding)
		knownnames.append(name)
print("processing unknown faces")
while True:
	ret, image = video.read()
	locations = face_recognition.face_locations(image, model=MODEL)
	encodings = face_recognition.face_encodings(image, locations)
	
	for faceencoding, facelocation in zip(encodings, locations):
		results = face_recognition.compare_faces(knownfaces, faceencoding, tolerance)
		match = None
		if True in results:
			match = knownnames[results.index(True)] 
			print(f"Match found: {match}")
			topleft = (facelocation[3], facelocation[0])
			bottomright = (facelocation[1], facelocation[2])
			color = [0, 255, 0]
			cv2.rectangle(image, topleft, bottomright, color, framethickness)
			topleft = (facelocation[3], facelocation[2])
			bottomright = (facelocation[1], facelocation[2]+22)
			cv2.rectangle(image, topleft, bottomright, color, cv2.FILLED)
			cv2.putText(image, match, (facelocation[3]+10, facelocation[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), fontthickness)
	cv2.imshow(filename, image)
	if cv2.waitKey(1) & 0xFF == ord("q"): 
		break
