import torch.backends.cudnn as cudnn

from PIL import Image
from borrower_tracker.settings import TRACKING_MODELS
from tracker.config import Config
from tracker.arcface.models import resnet_face18
from tracker.retinaface.detector import RetinafaceDetector
from tracker.utils.face_util import *

opt = Config()
opt.is_resize = True
opt.resize = 416
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

    height, width = img_raw.shape[:2]
    ratio = width / height
    if width >= height:
        is_width_larger = True
    else:
        is_width_larger = False
    if is_width_larger:
        # ratio > 1
        new_height = opt.resize
        new_width = int(ratio * new_height)
    else:
        # ratio < 1
        new_width = opt.resize
        new_height = int(new_width / ratio)

    if opt.is_resize:
        img_raw = cv2.resize(img_raw, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    img = img_raw.copy()
    # view image resized
    # cv2.imshow('fawef',img)
    # cv2.waitKey(0)
    det, patch = process(img, detector, output_size=(112, 112))

    print(det.shape, patch.shape)

    return patch


def save_features(img, name, uid):
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
    save_feat(opt, name, feat, TRACKING_MODELS.Faces)

    opt.features_path = origin_features_path


if __name__ == '__main__':
    src_path = './face.jpg'
    img_raw = cv2.imread(src_path)
    faces = face_recognition(img_raw)

    for face in faces:
        cv2.imshow('face', face)
        cv2.waitKey(0)
