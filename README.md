# SumDoNi
<B>To you who run the market.</B>

When a customer claims to have paid off, do you believe in the record or memory?
Can you stay calm and check CCTV in the situation he claims?
Wouldn't the distance between the customer and the customer become distant while checking the records?
For you in this situation, we will find the scene of a transaction with a customer who borrowed credit from CCTV.

<B>To you who are interested in deep learning</B>

You don't have CCTV, but are you interested in our repository? Don't worry, we also provide a demo that works with images or video without CCTV.

# Table of Contents

- #### [Download pretrained weights](#download-pretrained-model)
- #### [Face Recognition with Arcface, RetinaFace](#face-recognition-with-arcface-retinaface)
  - [Demo with Image or Video](#demo-with-image-or-video)
<br>

# Face Recognition with Arcface, RetinaFace 

![](https://github.com/Songminkee/Customer_face_recognition/blob/master/fig/demo.jpg)



<br>

# Download pretrained model

- Arcface & RetinaFace [[Google Drive]](https://drive.google.com/file/d/1-jjGFn6uoDHOl0OdIbOGYumgPjcRinIZ/view?usp=sharing)
  - [[Pytorch Arcface Repo]](https://github.com/ronghuaiyang/arcface-pytorch)
    - [[Finetune ArcFace Repo]](https://github.com/Songminkee/Asian-masked-arcface-pytorch)(our)
    - [[Download finetuned pretrained weight]](https://drive.google.com/file/d/1IbZs0uyLwibsjhhf37ZPf96BWwRSFg6N/view?usp=sharing)
  - [[Pytorch RetinaFace Repo]](https://github.com/biubug6/Pytorch_Retinaface)
    - [[Finetune RetinaFace Repo]](https://github.com/wooks527/Pytorch_Retinaface)(our)
    - [[Download finetuned pretrained weight]](https://drive.google.com/file/d/11jLXiN7zez9wEdXR3Y5V_vTXfgmRV_gl/view?usp=sharing)
- Deep sort & Yolov5 [[Google Drive]](https://drive.google.com/file/d/1wfyin2t_3kFFj2ENbJxDlLVPTUUr0EEe/view?usp=sharing)
  - Deep sort origin [[Web site]](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)
  - Yolov5 origin [[Github]](https://github.com/ultralytics/yolov5/releases)

<br>

### Demo with Image or Video

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

## Reference

- https://github.com/foamliu/Face-Alignment

- https://github.com/ronghuaiyang/arcface-pytorch

- https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch
