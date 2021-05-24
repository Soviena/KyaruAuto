# KyaruAuto
## _Princess Connect Re:Dive automation bot_

Still in development. Tested for Bluestack N-64@1280x720 with Debug bridge

## Features
- Auto dungeon (Hard)
- Auto optimize chara
- Buy all in bonus shop

## BUGS
- stuck in grotto
- daily() sometimes stuck in menu

## Roadmap
- Set Up party
- Better text recognition
- A GUI
- Installer
- Refactoring
- Auto Quest
- Auto Event

## Installation
KyaruAuto requires Python 3+, and teserract

Install the required python library.
```sh
pip install pure-python-adb
pip install imutils
pip install opencv-python
pip install numpy
pip install pytesseract
```


## Required pkg

| Name | Link |
| ------ | ------ |
| ADB | https://developer.android.com/studio/command-line/adb |
| Python | https://www.python.org/downloads/ |
| Tesseract-OCR | https://github.com/UB-Mannheim/tesseract/wiki |
| EAST (already included) | https://github.com/oyyd/frozen_east_text_detection.pb/blob/master/frozen_east_text_detection.pb |


## Add Python and Tesseract-OCR to PATH
- In Search, search for and then select: System (Control Panel)
- Click the Advanced system settings.
- Click Environment Variables. In the section System Variables find the PATH environment variable and select it. Click Edit. 
- If the PATH environment variable does not exist, click New.
- In the Edit System Variable (or New System Variable) window, specify the value of the PATH environment variable. Click OK. 
