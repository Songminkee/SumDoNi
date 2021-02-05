# Customer Face Recognition

- #### [Face Recognition with Arcface, RetinaFace](#face-recognition-with-arcface-retinaface)
  - [Inference example](#inference)

- #### [Download pretrained model](#download-pretrained-model)

<br>

# Face Recognition with Arcface, RetinaFace 

![](https://github.com/Songminkee/Customer_face_recognition/blob/master/fig/demo.jpg)



<br>

### Inference

- image mode

  ```
  python Inference.py --img_mode --src_path [your image file] --write --result_name result.jpg
  ```

- video mode

  ```
  python Inference.py --src_path [your video file] --write --result_name result.mp4
  ```

<br>

# Download pretrained model

- Arcface & RetinaFace [[Google Drive]](https://drive.google.com/file/d/1-jjGFn6uoDHOl0OdIbOGYumgPjcRinIZ/view?usp=sharing)

<br>

## Reference

- https://github.com/foamliu/Face-Alignment

- https://github.com/ronghuaiyang/arcface-pytorch