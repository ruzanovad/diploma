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

величина буквы зависит от шрифта

для начала достаточно template matching для подготовки bounding box

Проблема - не фиксированное количество объектов которые могут быть найдены 

Конечный автомат для состояний

Есть различные уровни обработки 

- числа целые
- числа десятичные
- символы
- *многочлены* - верхние и нижние индексы

сначала генерировать по уровням

нужно определить наименьшее множество шаблонов для получения  https://www.overleaf.com/learn/latex/Subscripts_and_superscripts

рассматриваем LaTeX как формальный язык

I deleted omicron and other greek letters because they doesn't exist in latex.

Как сгенерировать хороший датасет?

TODO генерировать файл о том, как часто встречается тот или иной класс в датасете
вообще нужно бы генерировать так, чтобы было равномерное распределение на множестве классов

map-50
Mean Average Precision


Intersection over Union (IoU): IoU is a measure that quantifies the overlap between a predicted bounding box and a ground truth bounding box. It plays a fundamental role in evaluating the accuracy of object localization.

Average Precision (AP): AP computes the area under the precision-recall curve, providing a single value that encapsulates the model's precision and recall performance.

Mean Average Precision (mAP): mAP extends the concept of AP by calculating the average AP values across multiple object classes. This is useful in multi-class object detection scenarios to provide a comprehensive evaluation of the model's performance.



Box(P, R, mAP50, mAP50-95): This metric provides insights into the model's performance in detecting objects:

    P (Precision): The accuracy of the detected objects, indicating how many detections were correct.

    R (Recall): The ability of the model to identify all instances of objects in the images.

    mAP50: Mean average precision calculated at an intersection over union (IoU) threshold of 0.50. It's a measure of the model's accuracy considering only the "easy" detections.

    mAP50-95: The average of the mean average precision calculated at varying IoU thresholds, ranging from 0.50 to 0.95. It gives a comprehensive view of the model's performance across different levels of detection difficulty.

https://docs.ultralytics.com/guides/yolo-performance-metrics/#class-wise-metrics

сначала надо делать датасет, а потом уже наращивать ...?   

ignoring corrupt image/label: image size (5, 5) <10 pixels

длинные изображения  в длину - 1300 пикселей, для обучения YOLO требует квадратные изображения поэтому проблематично 
использовать большой размер  

If custom floats are defined using a package like float are not supported
yet. Dependent on the way they define floats they might still work. For these
float=true should be set as class options so that the normal definition of floats
is preserved. Afterwards \standaloneconfig{float=false} can be used to
disable floats while taking the changed float definition into account.
convert={〈conversion options〉}
png={〈conversion options〉}
jpg={〈conversion options〉}
gif={〈conversion options〉}
svg={〈conversion options〉}
emf={〈conversion options〉}
These options allow to enable and configure the conversion feature. See section 6
for the full description.

если для 12 классов хватает 2000 изображений
то если добавляем буквы латинские (26*2=42)
если добавляем греческие буквы => получается порядка 100 классов
