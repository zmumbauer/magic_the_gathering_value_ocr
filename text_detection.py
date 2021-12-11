# Imports
from cv2 import cv2
import pdb
import numpy as np
import pytesseract as pt
import json
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import keyboard
import os
import time

pt.pytesseract.tesseract_cmd= "/usr/local/Cellar/tesseract/4.1.1/bin/tesseract"

def empty():
    pass

# Saves data to file
def save_to_file(data, setName):
    if os.path.exists(f"./collection/{setName}.json"):
        with open(f"./collection/{setName}.json", "r+", encoding='utf-8') as json_file:
            existingData = json.load(json_file)
            if data['name'] in existingData:
                existingData[data['name']].update({"quantity": existingData[data['name']]['quantity'] + 1})
            else:
                existingData.update({data['name']: {
                    "value": data['value'],
                    "quantity": 1
                }})
            json_file.seek(0)
            json.dump(existingData, json_file, ensure_ascii=False, indent=4)
    else:
        with open(f"./collection/{setName}.json", "w", encoding='utf-8') as json_file:
            json.dump({data['name']: {
                "value": data['value'],
                "quantity": 1
            }}, json_file, ensure_ascii=False, indent=4)

# Pulls a list of sets from the dataset
def user_select_set():
    with open('data.json', 'r') as json_file:
        data = json.load(json_file)
    setList = []
    for d in data:
        setList.append(d)

    # Get set
    print('What set are you looking for?')
    setQuery = input()
    filteredSets = list(filter(lambda set: fuzz.token_sort_ratio(setQuery, set) > 70, setList))
    selectedSet = ""
    
    # If more than one set matches search, select index
    if len(filteredSets) > 1:
        print("One of these? (Select a number)")
        for num, set in enumerate(filteredSets):
            print(f"{num}. {set}")

        while(selectedSet == ""):
            index = int(input())
            if index < len(filteredSets) and index >= 0:
                selectedSet = filteredSets[index]
                break
            else: 
                pass
        
    else:
        selectedSet = filteredSets[0]
    return selectedSet

# Pulls a list of cards from the dataset
def build_card_list(setName):
    with open('data.json', 'r') as json_file:
        data = json.load(json_file)
    cardList = data[setName]['cards']
    return cardList

# Returns the card dictionary if it fuzzy matches key in database
def get_card_by_name(cardList, name):
    # Filter cards by fuzzy matching name at 90 percent
    cards = list(filter(lambda card: fuzz.token_sort_ratio(name, card["name"]) > 85, cardList))
    if len(cards) > 0:
        # Return the first card in the list
        return sorted(cards, key = lambda card: card["value"], reverse=True)[0]
    return None

# Finds the border of the card and returns the coordinates for the corners
def get_contours(img):
    maxContour = np.array([])
    maxArea = 0
    contours, heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, .02 * perimeter, True)
            if area > maxArea and len(approx) == 4:
                maxContour = approx
                maxArea = area
                # cv2.drawContours(processedFrame, maxContour, -1, (255, 0, 0), 30)
                cv2.drawContours(output, maxContour, -1, (255, 0, 0), 30)
    return maxContour

def process_image(img):
    greyed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(greyed, (5,5), 0)
    edged = cv2.Canny(blured, 100, 100)
    # kernel = np.ones((5, 5))
    # dialated = cv2.dilate(edged, kernel, iterations=2)
    # threshold = cv2.erode(dialated, kernel, iterations=1)
    return edged

def reorder_points(documentEdge):
    points = documentEdge.reshape((4, 2))
    newPoints = np.zeros((4, 1, 2), np.int32)

    # Sums the x and y for each coordinate and adds to sum array
    sum = points.sum(1)
    newPoints[0] = points[np.argmin(sum)]
    newPoints[3] = points[np.argmax(sum)]

    diff = np.diff(points, axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]

    return newPoints

def warp_image(originalImg, img, documentEdge):
    width, height = img.shape
    if len(documentEdge) == 4:
        x,y,w,h = cv2.boundingRect(documentEdge)
        # Crops the image to just the card
        card = originalImg[y:y+h, x:x+w]
        documentEdge = reorder_points(documentEdge)

        pts1 = np.array(documentEdge)
        pts2 = np.array([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
        warpedImage = cv2.warpPerspective(originalImg, matrix, (w, h))
        return warpedImage
    else:
        return img

def card_title_ocr(img):
    print(pt.image_to_string(img))
    cv2.imshow("TitleIMG", img)
    return pt.image_to_string(img, lang="eng", config="--oem 1 --psm 6")

def text_gen(img, string, position, size):
    cv2.putText(img, string, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), size, cv2.LINE_AA)

def handle_card_found(img, card, documentEdge):
    # Adds card name and value to frame output
    x, y, w, h = cv2.boundingRect(documentEdge)
    cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.rectangle(output, (x, y+h), (x+w, y+h + 100), (255, 0, 0), -1)
    text_gen(img, card['name'], (x+10, y+h+40), 2)
    text_gen(img, card['value'], (x+10, y+h+80), 2)
    render_frame()
    # Pause feed
    cv2.waitKey(0)
    # If the card has any value ask to save
    if float(card["value"][1:]) > 0:
        print(f"{card['name']} ({card['value']}) detected. Press spacebar to save any other key to discard.")
        while(True):
            _, _, width, height = cv2.getWindowImageRect("MTG Value Finder")
            key = keyboard.read_key()
            if key == 'space':
                save_to_file(card, setName)
                print(f"{card['name']} saved to file")
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), -1)
                text_gen(output, "Saved", (x+int(w/2) - 40, y+int(h/2)), 2)
                render_frame()
                cv2.waitKey(2)
                break
            elif key == 'd':
                print(f"Discarded {card['name']}")
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), -1)
                render_frame()
                cv2.waitKey(2)
                break

def calculate_collection_value():
    cards = {}
    # Get all cards
    for file in os.listdir('./collection/'):
        print(f"Processing {file}")
        with open(f"./collection/{file}", 'r') as json_file:
            cards.update(json.load(json_file))
    
    # Calculate aggregate value
    value = 0
    for name, price in cards.items():
        value += float(price['value'][1:])
    return value

def render_frame():
    cv2.imshow("MTG Value Finder", output)

# User selects a set
setName = user_select_set()
print(f"{setName} set is selected")

# Get card list from data
cardList = build_card_list(setName)

# Init webcam connection
cv2.namedWindow("MTG Value Finder")
camera = cv2.VideoCapture(2)
retVal, frame = camera.read()
camera.set(3, 1080)
camera.set(4, 1920)

# Read camera and display to screen until user presses 'q' key
while camera.isOpened():
    # Read frame from webcam stream
    retVal, frame = camera.read()

    # Rotate frame 90 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Create output image
    output = frame.copy()

    # Process the frame
    processedFrame = process_image(frame)

    # Get the bounding box of the card
    documentEdge = get_contours(processedFrame)

    # Use warped translate 
    warpedImg = warp_image(frame, processedFrame, documentEdge)

    # Get name of card from ocr on cropped image
    cardTitleOCR = card_title_ocr(warpedImg[0:80, 0:300])
    card = get_card_by_name(cardList, cardTitleOCR)

    # Show frame in window
    render_frame()
    
    # If the card is found
    if card != None:
        handle_card_found(output, card, documentEdge)
            
    # Wait for user to press 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
