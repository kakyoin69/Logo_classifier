from loguru import logger
from src import ImageUtils
from src.classificators.clip import Clip
from src.classificators.effiiceint import Efficientnet


class ProcessManager:
   def __init__(self, config: dict):
      self.config = config
      self.image_utils = ImageUtils()
      self.image = self.image_utils.open_image(self.config['image'])
      self.images = self.image_utils.open_images(self.config['images'])
      self.text = self.config['text']
      self.labels = [self.text, f'other logo']
      
      if self.config['weights'] and self.config['classes']:
         self.classifier_eff = Efficientnet(model_path=self.config['weights'], classes_file=self.config['classes'])
      if self.text:
         self.classifier_clip = Clip(labels=self.labels)


   def worker(self):
      answer = {}
      if self.image and self.images:
         sim_list_path = []
         anchor_cls = self.classifier_eff.predict_image_class(self.image)
         for num_img in range(0, len(self.images)):
            tmp_cls = self.classifier_eff.predict_image_class(self.images[num_img])
            if tmp_cls == anchor_cls:
               sim_list_path.append(self.config['images'][num_img])
         answer['sim'] = {anchor_cls : sim_list_path}    
      
      if self.text and self.image:
         result = self.classifier_clip.classify_image(image=self.image)
         best_key = max(result, key=result.get)
         best_value = result[best_key]
         if best_key == self.text:
            answer['brand'] = "Yes, this is the correct logo."
         else:
            answer['brand'] = "No, this is a different logo."
      return answer
