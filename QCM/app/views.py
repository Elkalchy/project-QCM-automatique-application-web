from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

fs = FileSystemStorage()

def index(request):
    reference_paths = []  # Liste pour stocker les chemins des fichiers de référence
    control_paths = []    # Liste pour stocker les chemins des fichiers de contrôle
    all_paths = []        # Liste pour stocker tous les chemins des fichiers

    if request.method == 'POST':
        # Récupère la liste des fichiers pour chaque champ
        reference_files = request.FILES.getlist('references')
        control_files = request.FILES.getlist('controles')

        # Enregistre chaque fichier de référence et ajoute le chemin dans `reference_paths`
        for reference_file in reference_files:
            reference_name = fs.save(reference_file.name, reference_file)
            reference_path = fs.path(reference_name).replace("\\", "/")  # Remplacement de '\' par '/'
            reference_paths.append(reference_path)

        # Enregistre chaque fichier de contrôle et ajoute le chemin dans `control_paths`
        for control_file in control_files:
            control_name = fs.save(control_file.name, control_file)
            control_path = fs.path(control_name).replace("\\", "/")  # Remplacement de '\' par '/'
            control_paths.append(control_path)

        # Combine les chemins de références et de contrôle dans un seul tableau
        all_paths = reference_paths + control_paths
        print("Tous les chemins de fichiers :", all_paths)

        # Traiter chaque combinaison d'images
        resultats = []
        try:
            for control_path, reference_path in zip(control_paths, reference_paths):
                resultat = process_operation(photo=control_path, photo2=reference_path)
                
                # Vérifier si `resultat` est bien un tuple avec 2 éléments
                if isinstance(resultat, tuple) and len(resultat) == 2:
                    resultats.append({'names': os.path.basename(resultat[1]), 'resultat': resultat[0]})
                else:
                    resultats.append({'names': 'Valeur par défaut', 'resultat': 'Erreur'})

            return render(request, 'index.html', {'resultats': resultats, 'all_paths': all_paths})
        except Exception as e:
            print("Erreur dans process_operation:", e)
            return render(request, 'index.html', {'error': 'Une erreur est survenue lors du traitement des images.'})

    return render(request, 'index.html')

def lire_qr_code(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Erreur : Impossible d'ouvrir ou de lire le fichier. Vérifie le chemin d'accès.")
        return

    detector = cv2.QRCodeDetector()
    data_control, points, _ = detector.detectAndDecode(img)

    if data_control:
        if points is not None:
            points = points[0].astype(int)
            for i in range(4):
                cv2.line(img, tuple(points[i]), tuple(points[(i + 1) % 4]), (0, 255, 0), 3)
        cv2.destroyAllWindows()
        return data_control
    else:
        print("Aucun QR Code trouvé.")
        return None

def process_operation(photo, photo2):
    heightImg = 700
    widthImg = 700
    questions = 5
    choices = 5

    correct_answers = extract_answers_from_image(image_path=photo2, questions=questions, choices=choices)
    print("Réponses correctes :", correct_answers)

    if photo:
        pathImage = photo
        print(f"Traitement de l'image de contrôle : {pathImage}")
        
        data_control = lire_qr_code(image_path=pathImage)
        data = extrat_code_qr_from_image(image_path=photo2)

        if data == data_control:
            img = cv2.imread(pathImage)
            img = cv2.resize(img, (800, 800))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            texte_extrait = "Non détecté"
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 100 and h > 20 and y < 200:
                    roi = img[y:y+h, x:x+w]
                    texte_extrait = pytesseract.image_to_string(roi, lang='eng')
                    print("Texte détecté :", texte_extrait)

            try:
                imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
                imgCanny = cv2.Canny(imgBlur, 10, 70)
                contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                rectCon = rectContour(contours)

                if len(rectCon) > 1:
                    biggestPoints = getCornerPoints(rectCon[0])
                    gradePoints = getCornerPoints(rectCon[1])

                    if biggestPoints.size != 0 and gradePoints.size != 0:
                        biggestPoints = reorder(biggestPoints)
                        pts1 = np.float32(biggestPoints)
                        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
                        matrix = cv2.getPerspectiveTransform(pts1, pts2)
                        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

                        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
                        imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

                        boxes = splitBoxes(imgThresh)
                        myPixelVal = np.zeros((questions, choices))

                        for countR in range(questions):
                            for countC in range(choices):
                                totalPixels = cv2.countNonZero(boxes[countR * choices + countC])
                                myPixelVal[countR][countC] = totalPixels

                        myIndex = [np.argmax(myPixelVal[x]) for x in range(questions)]
                        grading = [1 if correct_answers[x] == myIndex[x] else 0 for x in range(questions)]
                        score = (sum(grading) / questions) * 20
                        print(f"Score pour {texte_extrait}: {score}")
                        return score, texte_extrait

            except Exception as e:
                print("Erreur lors du traitement : ", e)

    return 0, "Aucun score calculé"

def extrat_code_qr_from_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Erreur : Impossible d'ouvrir ou de lire le fichier. Vérifie le chemin d'accès.")
        return None

    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(img)

    if data:
        if points is not None:
            points = points[0].astype(int)
            for i in range(4):
                cv2.line(img, tuple(points[i]), tuple(points[(i + 1) % 4]), (0, 255, 0), 3)
        cv2.destroyAllWindows()
        return data
    else:
        print("Aucun QR Code trouvé.")
        return None

def extract_answers_from_image(image_path, questions=5, choices=5):
#---------------------------------traitement pour detecter les reponses ---------------------

    # Lire et redimensionner l'image
    img = cv2.imread(image_path)
    heightImg = 700
    widthImg = 700
    img = cv2.resize(img, (widthImg, heightImg))

    # Prétraitement de l'image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 70)

    # Trouver tous les contours
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rectCon = rectContour(contours)

    # Si les contours rectangulaires sont trouvés
    if len(rectCon) > 0:
        biggestPoints = getCornerPoints(rectCon[0])  # Plus grand rectangle

        if biggestPoints.size != 0:
            # Transformation de perspective
            biggestPoints = reorder(biggestPoints)
            pts1 = np.float32(biggestPoints)
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            # Appliquer un seuil pour isoler les cases
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

            # Diviser en cases
            boxes = splitBoxes(imgThresh)
            myPixelVal = np.zeros((questions, choices))

            countR = 0
            countC = 0
            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixels
                countC += 1
                if countC == choices:
                    countC = 0
                    countR += 1

            # Trouver les réponses correctes
            ans = []
            for x in range(0, questions):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.amax(arr))
                ans.append(myIndexVal[0][0])  # Ajouter l'indice de la réponse la plus remplie

            return ans
    return []

def rectContour(contours):
    rectCon = []
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    return rectCon

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
    return approx

def splitBoxes(img):
    rows = np.vsplit(img, 5)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 5)
        for box in cols:
            boxes.append(box)
    return boxes

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]  # Top-left
    myPointsNew[3] = myPoints[np.argmax(add)]  # Bottom-right

    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # Top-right
    myPointsNew[2] = myPoints[np.argmax(diff)]  # Bottom-left
    return myPointsNew

