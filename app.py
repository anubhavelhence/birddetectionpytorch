from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
from torchvision import transforms
import 
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


# Initialize your model here
model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True,
                       **{'topN': 6, 'device':'cpu', 'num_classes': 200})
model.eval()

# Image transformation
transform_test = transforms.Compose([
    transforms.Resize((600, 600), Image.BILINEAR),
    transforms.CenterCrop((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_url = request.json['url']
    img = Image.open(urllib.request.urlopen(img_url))
    scaled_img = transform_test(img)
    torch_images = scaled_img.unsqueeze(0)

    with torch.no_grad():
        _, _, _, concat_logits, _, _, _ = model(torch_images)
        _, predict = torch.max(concat_logits, 1)
        pred_id = predict.item()
        
    return jsonify({'bird_class': model.bird_classes[pred_id]})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)