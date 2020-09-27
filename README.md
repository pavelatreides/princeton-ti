# princetontechinterview

This is the link for the solutions to the technical challenge posited by Princeton University for the position of Research Software Engineer. Here you fill find the directories and files necessary to run Task A. 

The widget was developed entirely in Python3/OS X using a Pyside2 application using ZMQ

![task-a-demo.gif](task-a-demo.gif)

The directory structure exists to faciliate the import and loading of provided files. Other folders and files are accessible for import/load, but these folders exist to enforce a default path when opening dialog windows. 'models' is where .h5 model files are stored, 'inputdata' is where videos are kept, and 'groundtruth' keeps the tracking information. I decided not to upload all the files you provided me to save space and cut down on difficult pushes to Git. Feel free to drop the appropriate files in the folders when you clone the repo.

## Installation
1. Clone this repo
2. `pip3 install -r requirements.txt`
  
  I highly recommend the creation of a virtual environment via to keep all the packages together
  ```
  python3 -m venv /path/to/new/virtual/environment
  ```

## Usage
Assuming all dependencies are installed, execution is as easy as:
```
cd task_a
python3 task_a.py
```
This will load (~15 sec) and run the GUI, which will create the server in a separate process and run the client.

## Troubleshooting
If you experience any issues getting the code to execute, please feel free to reach out to me at pavel.ser.isakov@gmail.com
If you experience any errors regarding Qt frameworks, it likely has to do with conflicting versions of Qt in OpenCV and Pyside2. This worked for me and might work for you:
```
pip3 install --no-binary opencv-python opencv-python
```

## Contributing
Developed solely by Pavel Isakov.

## License
None necessary here. Private repo for interviewing only.
