DIRECTORY="models"

if [ ! -d "$DIRECTORY" ]; then
  mkdir $DIRECTORY
fi

cd $DIRECTORY

wget  http://dlib.net/files/mmod_human_face_detector.dat.bz2
bunzip2 mmod_human_face_detector.dat.bz2

wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
	
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt
