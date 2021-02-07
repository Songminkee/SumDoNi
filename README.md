# Customer Face Recognition

- #### [Face Recognition with Arcface, RetinaFace](#face-recognition-with-arcface-retinaface)
  - [Inference example](#inference)

- #### [Download pretrained weights](#download-pretrained-model)

<br>

# Face Recognition with Arcface, RetinaFace 

![](https://github.com/Songminkee/Customer_face_recognition/blob/master/fig/demo.jpg)



<br>

### Inference

- image mode

  ```
  python demo_face_recognition.py --img_mode --src_path [your image file] --write --result_name result.jpg
  ```

- video mode

  ```
  python demo_face_recognition.py --src_path [your video file] --write --result_name result.mp4
  ```

- tracking mode

  ```
  python demo_person_tracker.py --source inputs/1.mp4
  ```

<br>

# Download pretrained model

- Arcface & RetinaFace [[Google Drive]](https://drive.google.com/file/d/1-jjGFn6uoDHOl0OdIbOGYumgPjcRinIZ/view?usp=sharing)
- Deep sort & Yolov5 [[Google Drive]](https://drive.google.com/file/d/1wfyin2t_3kFFj2ENbJxDlLVPTUUr0EEe/view?usp=sharing)
  - Deep sort origin [[Web site]](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)
  - Yolov5 origin [[Github]](https://github.com/ultralytics/yolov5/releases)

<br>

## Reference

- https://github.com/foamliu/Face-Alignment

- https://github.com/ronghuaiyang/arcface-pytorch

- https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch
