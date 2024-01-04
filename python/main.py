import cv2
import onnxruntime as ort
import numpy as np
from tokenizer import build_tokenizer
print(ort.__version__)
class OWLVIT():
    def __init__(self, image_modelpath, text_modelpath, post_modelpath, box_thresh = 0.2, text_thresh = 0.25):
        self.image_model = cv2.dnn.readNet(image_modelpath)
        self.input_height, self.input_width = 768, 768
        
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073],
                             dtype=np.float32).reshape((1, 1, 3))
        self.std = np.array([0.26862954, 0.26130258, 0.27577711],
                            dtype=np.float32).reshape((1, 1, 3))
        
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.bert = ort.InferenceSession(text_modelpath, so)
        self.bert_input_names = []
        for i in range(len(self.bert.get_inputs())):
            self.bert_input_names.append(self.bert.get_inputs()[i].name)

        self.transformer = ort.InferenceSession(post_modelpath, so)
        self.transformer_input_names = []
        for i in range(len(self.transformer.get_inputs())):
            self.transformer_input_names.append(self.transformer.get_inputs()[i].name)
        
        self.box_thresh = box_thresh
        self.text_thresh = text_thresh
        self.tokenizer = build_tokenizer('bpe_simple_vocab_16e6.txt.gz')
    
    def preprocess(self, srcimg):
        img = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        img = (img.astype(np.float32)/255.0 - self.mean) / self.std
        return img
    
    def encode_image(self, srcimg):
        img = self.preprocess(srcimg)
        blob = cv2.dnn.blobFromImage(img)
        self.image_model.setInput(blob)
        image_features, pred_boxes = self.image_model.forward(self.image_model.getUnconnectedOutLayersNames())
        return image_features, pred_boxes.reshape(-1,4)
    
    def encode_texts(self, text_prompt):
        token_ids = [self.tokenizer.encode(t) for t in text_prompt]
        input_ids, text_features = [], []
        for ids in token_ids:
            input_id = np.pad([49406, *ids, 49407],(0,16-len(ids)-2)).astype(np.int64)
            input_ids.append(input_id)
            mask = (input_id > 0).astype(np.int64)

            text_feature = self.bert.run(None, {self.bert_input_names[0]:input_id.reshape(1,16), self.bert_input_names[1]:mask.reshape(1,16)})[0].reshape(1,-1)
            text_features.append(text_feature)
        return text_features, input_ids
    
    def decode(self, image_feature, text_feature, input_id):
        logits = self.transformer.run(None, {self.transformer_input_names[0]:image_feature[0].reshape(1,24,24,768), self.transformer_input_names[1]:text_feature, self.transformer_input_names[2]:input_id.reshape(1,16)})[0]
        logits = 1/(1+np.exp(-logits)).reshape(-1)  ###sigmoid
        return logits

    def detect(self, srcimg, text_prompt):
        if isinstance(text_prompt, str):
            text_prompt = [text_prompt]
        srch, srcw = srcimg.shape[:2]
        image_features, pred_boxes = self.encode_image(srcimg)
        text_features, input_ids = self.encode_texts(text_prompt)
        objects = []
        for i,input_id in enumerate(input_ids):
            logits = self.decode(image_features, text_features[i], input_id)
            boxes = pred_boxes[logits > self.box_thresh]  ###形状nx4
            score = logits[logits > self.box_thresh]
            for j in range(boxes.shape[0]):
                #cx,cy,w,h = boxes[j, :]
                xmin = int((boxes[j, 0]-0.5*boxes[j, 2])*srcw)
                ymin = int((boxes[j, 1]-0.5*boxes[j, 3])*srch)
                xmax = int((boxes[j, 0]+0.5*boxes[j, 2])*srcw)
                ymax = int((boxes[j, 1]+0.5*boxes[j, 3])*srch)
                objects.append({'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax,'name':text_prompt[i],'score':score[j]})
        return objects
    
if __name__=='__main__':
    mynet = OWLVIT('weights/owlvit-image.onnx', 'weights/owlvit-text.onnx', 'weights/owlvit-post.onnx')

    imgpath = 'images/test.jpg'
    srcimg = cv2.imread(imgpath)
    text_prompt = ["football", "a photo of person"]  ###人，不能直接写person,要写成a photo of person
    
    objects = mynet.detect(srcimg, text_prompt)

    for obj in objects:
        cv2.rectangle(srcimg, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (0,0,255), 2)
        cv2.putText(srcimg, obj['name'], (obj['xmin'], obj['ymin']-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

    # cv2.imwrite('result.jpg', drawimg)
    winName = 'Simple Open-Vocabulary Object Detection with Vision Transformers use OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()