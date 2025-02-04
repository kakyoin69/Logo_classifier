from loguru import logger

class DataLoader:
    def __init__(self, config: dict, args):
        if config and (config.get('images') or config.get('text') or config.get('image')
                        or config.get('weights') or config.get('classes')):
            self.images = config.get('images')
            self.text = config.get('text')
            self.image = config.get('image')
            self.weights = config.get('weights')
            self.classes = config.get('classes')
        else:
            self.images = args.images
            self.text = args.text
            self.image = args.image
            self.weights = args.weights
            self.classes = args.classes

    def validate_inputs(self):
        if not any([self.images, self.text, self.image]):
            raise ValueError(
                "No input paths provided. Please specify at least one of the following: images, text, or image."
            )

    def load_data(self) -> dict:
        self.validate_inputs()
        return {
            'images': self.images,
            'text': self.text,
            'image': self.image,
            'weights': self.weights,
            'classes': self.classes
        }