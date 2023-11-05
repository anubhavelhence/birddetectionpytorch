from flask import Flask, render_template, request, jsonify, redirect, url_for,render_template_string, abort
import torch
from PIL import Image
from torchvision import transforms
import urllib
from flask_cors import CORS
from pymongo import MongoClient
import requests
import os
from flask_httpauth import HTTPDigestAuth






app = Flask(__name__)
CORS(app)



# Replace with your actual API key
API_KEY = 'DEADC0DE'



# Get username, password, and database name from environment variables for security
# MONGO_USER = os.getenv('MONGO_USER')
# MONGO_PASS = os.getenv('MONGO_PASS')
# MONGO_DB = os.getenv('MONGO_DB')

client = MongoClient("mongodb://yourNewUsername:yourNewPassword@localhost:27017/")
db = client['bird_database']
bird_collection = db['birds']

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

@app.route('/bird_classes')
def bird_classes():
    birds = list(bird_collection.find({}))
    
    return render_template('bird_classes.html', birds=birds)

@app.route('/debug_route')
def debug_route():
    # return render_template('debug_route.html')
    url = "https://ebird.org/species/layalb" # Replace with the URL you want to capture
    response = requests.get(url)
    page_content = response.text

    return render_template_string('<html><body>{{ content|safe }}</body></html>', content=page_content)


@app.route('/new_bird', methods=['GET', 'POST'])
def new_bird():
    if request.method == 'POST':
        # class_number = int(request.form.get('classNumber'))
        class_number = request.form.get('classNumber')

        species_name = request.form.get('species')
        species_code = request.form.get('speciesCode')
        species_url = request.form.get('speciesUrl')
        species_photo = request.form.get('speciesPhoto')
        loc = request.form.get('loc')


        bird_data = {
            'ClassNumber': class_number,
            'Species': species_name,
            'Speciescode': species_code,
            'SpeciesUrl': species_url,
            'SpeciesPhoto': species_photo,
            'Loc': loc
        }

        print(bird_data)

        result = bird_collection.insert_one(bird_data)

        # Verify if the document was inserted
        if result.acknowledged:
            print(f"Successfully inserted document with _id: {result.inserted_id}")
        else:
            print("Failed to insert document")
        return redirect(url_for('bird_classes'))

    return render_template('new_bird.html')

@app.route('/get_bird_metadata', methods=['GET'])
def get_bird_metadata():
    species_name = request.args.get('species')
    
    print(species_name)
    bird = bird_collection.find_one({"Species": species_name})
    print(bird)
    if bird:
        bird_dict = dict(bird)  # Convert to Python dictionary
        bird_dict.pop('_id', None)  # Remove the _id field
        print(bird_dict)
        return jsonify(bird_dict)  # Serialize and send as JSON
    else:
        return jsonify({"error": "Bird not found"}), 404


@app.route('/capture', methods=['GET'])
def capture():
    speciescode = request.args.get('speciescode')
    url = "https://ebird.org/species/" + str(speciescode)
    print(url)
    response = requests.get(url)
    page_content = response.text
    return jsonify({"content": page_content, "url":url})

@app.route('/data', methods=['GET'])
def get_data():
    # Extract the API Key from the request's headers
    api_key = request.headers.get('x-api-key')
    
    # Check if the API key is valid
    if api_key != API_KEY:
        abort(401)  # Unauthorized access
    
    # Example data to return
    data = {
        'message': 'Access granted. Here is your data!',
        'data': [1, 2, 3, 4, 5]
    }
    
    return jsonify(data), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)