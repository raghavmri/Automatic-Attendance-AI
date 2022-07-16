import cv2
import face_recognition

trainImg = face_recognition.load_image_file("images/prod.jpeg")
trainImg = cv2.cvtColor(trainImg, cv2.COLOR_BGR2RGB)

prodImg = face_recognition.load_image_file("images/test.jpeg")
prodImg = cv2.cvtColor(prodImg, cv2.COLOR_BGR2RGB)


faceLoc = face_recognition.face_locations(prodImg)[0]
faceEncdoing = face_recognition.face_encodings(prodImg)[0]


faceLocTest = face_recognition.face_locations(trainImg)[0]
faceEncdoingTest = face_recognition.face_encodings(trainImg)[0]


res = face_recognition.compare_faces([faceEncdoing],faceEncdoingTest)
ress = face_recognition.face_distance([faceEncdoing],faceEncdoingTest)
print(res)
print(ress)
# cv2.rectangle(prodImg,(faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]),(255,0,255),2)
# cv2.rectangle(trainImg,(faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]),(255,0,255),2)

# cv2.imshow("Training Image",trainImg)
# cv2.imshow("Prod Image",prodImg)
# cv2.waitKey(0)
