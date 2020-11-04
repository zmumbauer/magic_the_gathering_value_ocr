# MTG OCR

## About

This project uses python, opencv, and tesseract automatically fetch the value of Magic the Gathering cards. It uses a video stream (external webcam by default) to detect documents with four edges, warps them into perspective, uses OCR to convert the card name to a string, then looks the value up in a prepopulated database. It displays the value on the feed and asks user if they want to save the card to a collection.

## Notes
Due to possible legal ramifications I did not include the web scraper or the dataset I made for this project.