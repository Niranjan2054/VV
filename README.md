# Deep Sort And YOLO V4

## installation Required
To run the project, python3.6 or above required and pip3 with latest version 20.X.X

upgrade the pip3 version to latest version...(Required)
```bash
sudo -H pip3 install --upgrade pip
```

To run  the required libraries run the install.sh file as:
```bash
./install.sh
```

## inclusion needed (Optional since file is already converted and stored in model_data)
[Download]('https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view') and add yolov4.weights file in model_data folder, then run 
```python
python3 convert.py 
```
This will convert the yolov4 weights file to keras model (.h5 file) 
The keras model will save in the model_data directory..

## inference 
```python
python3 main.py
```

change video name on line 56 in main.py
```bash
    file_path = '[filename].[ext]'
```
