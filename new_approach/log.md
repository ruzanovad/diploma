# Написана функция создания паттернов цифр
# YOLO v8, задача сегментации
# Roboflow для разметки объектов
# Одним из наивных подходов на основе свёрточных нейронных сетей может быть использование в качестве ядра каноничных изображений классов, которые необходимо найти на изображении, и дальнейшее использование скользящего окна для вычисления свёртки. Такой подход называется сопоставлением с шаблоном (англ. template matching).
[https://amslaurea.unibo.it/id/eprint/26836/1/Convolutional%20Neural%20Network%20Architectures%20for%20Template%20Matching.pdf](https://amslaurea.unibo.it/id/eprint/26836/1/Convolutional%20Neural%20Network%20Architectures%20for%20Template%20Matching.pdf)

---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
Cell In[2], line 23
     20 for pt in zip(*loc[::-1]):
     21     cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
---> 23 cv2.imshow('Detected', image)
     24 cv2.waitKey(0)
     25 cv2.destroyAllWindows()

error: OpenCV(4.10.0) /io/opencv/modules/highgui/src/window.cpp:1301: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'

on fedora ```sudo dnf install gtk3-devel``

# DVC

```dvc init --subdir```
```dvc add number_patterns dataset```

Template matching could not be suitable

Fixed Size vs. Variable Size: Do all patterns have the same size?
Fixed Position vs. Random Position: Are patterns always in the same place within the image?
Simple vs. Complex Background: Do they appear on a uniform background, or is it complex?
Limited vs. Multiple Patterns: Are there multiple pattern classes?

Требуется мануальная разметка 

pip install labelImg
dvitype for getting text from dvi

for patterns it is easy because patterns bounding box is a whole picture width and height