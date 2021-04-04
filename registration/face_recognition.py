import torch.backends.cudnn as cudnn

from PIL import Image
from tracker.config import Config
from tracker.arcface.models import resnet_face18
from tracker.retinaface.detector import RetinafaceDetector
from tracker.utils.face_util import *

opt = Config()
opt.is_resize = True
opt.resize = 256
opt.indivisual_threshold = False


class ModelClass:
    def __init__(self):
        self.arc_model = resnet_face18(opt.use_se)
        load_model(self.arc_model, opt.arc_model_path)
        cudnn.benchmark = True
        device = torch.device("cuda")
        self.arc_model.to(device)
        self.arc_model.eval()

        self.detector = RetinafaceDetector(is_gpu=True, opt=opt)

    def get_arc_model(self):
        return self.arc_model

    def get_detector(self):
        return self.detector


set_model = ModelClass()


def face_recognition(img_raw):
    detector = set_model.get_detector()

    # Image
    img_raw = np.asarray(img_raw)
    if opt.is_resize:
        img_raw = cv2.resize(img_raw, (opt.resize, opt.resize), interpolation=cv2.INTER_LINEAR)
    img = img_raw.copy()
    det, patch = process(img, detector, output_size=(112, 112))

    print(det.shape, patch.shape)

    return patch


def save_features(img, name, uid):
    Faces = FaceFeatures(opt.features_path)
    origin_features_path = opt.features_path
    opt.features_path = opt.features_path + '/' + str(uid)

    if not os.path.exists(os.path.join(opt.features_path,name)):
        os.makedirs(os.path.join(opt.features_path,name))
    exists = glob.glob(os.path.join(opt.features_path,name)+'/*.jpg')
    dst_path = f'{opt.features_path}/{name}/{name}_{len(exists)+1}.jpg'
    if img is not None:
        img_ = Image.fromarray(img)
        img_.save(dst_path, "JPEG")

    arc_model = set_model.get_arc_model()
    p = cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2RGB)
    feat = get_face_feature(arc_model, cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), preprocess=True)
    save_feat(opt, name, feat, Faces)

    opt.features_path = origin_features_path


if __name__ == '__main__':
    src_path = './face.jpg'
    img_raw = cv2.imread(src_path)
    faces = face_recognition(img_raw)

    for face in faces:
        cv2.imshow('face', face)
        cv2.waitKey(0)
