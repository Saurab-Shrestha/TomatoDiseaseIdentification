import os
import pandas as pd
from flask import Flask, render_template, request, url_for, redirect
from PIL import Image
import torch
from torchvision import transforms

disease_df = pd.read_csv("Web app/disease.csv")

# Define a flask app
app = Flask(__name__)
# load your trained model
device = torch.device('cpu')

from model.final_model import CNNModel

# actual tomato leaf disease CNN model
export_file = 'Web app/model/final_model.pth'
model = CNNModel()
model.load_state_dict(torch.load(export_file,map_location=device))

# model to classify whether the image is tomato or not (binary classifier)
tomatoornot_file = 'Web app/model/tomatoornot.pth'
tomatoornot_model = CNNModel()
tomatoornot_model.load_state_dict(torch.load(tomatoornot_file,map_location=device))

idx_to_class = {
    0: 'Not Tomato', 
    1: 'Tomato'
}

def predict_tomato_or_not(img):
    image = Image.open(img).convert('RGB')

    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)

    # Pass through the model
    tomatoornot_model.eval()
    with torch.no_grad():
        #image = image # Move input tensor to the GPU
        output = tomatoornot_model(image.unsqueeze(0))  # Add batch dimension
    predicted_class = torch.argmax(output, dim=1).item()
    actual_class = idx_to_class[predicted_class]
    print('Predicted class:', actual_class)
    return actual_class

def predict_disease(img):
    image = Image.open(img).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = transform(image)

    # Pass through the model
    model.eval()
    with torch.no_grad():
        #image = image # Move input tensor to the GPU
        output = model(image.unsqueeze(0))  # Add batch dimension
    predicted_probs = torch.softmax(output, dim=1)
    print(predicted_probs)
    
    predicted_class = torch.argmax(output, dim=1).item()
    actual_class = disease_df["Disease_name"][predicted_class]

    confidence = predicted_probs[0][predicted_class].item()
    confidence = confidence * 100
    confidence = float("{:.2f}".format(confidence))

    desc = disease_df["Description"][predicted_class]
    prev = disease_df['Prevention'][predicted_class]
    desc = disease_df["Description"][predicted_class]
    prev = disease_df['Prevention'][predicted_class]

    # Print the predicted class and confidence
    print('Predicted class:', actual_class)
    print('Confidence: ',confidence)
    print("description: ",desc)
    print("Prevention: ",prev)

    return actual_class, confidence, desc, prev

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/submit',methods=['GET','POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('Web app/static/uploads/', filename)
        image.save(file_path)
        print(file_path)
 

        tomatoornot = predict_tomato_or_not(file_path)
        if tomatoornot == 'Tomato':
            pred, confidence,desc, prev = predict_disease(file_path)
            image_url = url_for('static', filename=f'uploads/{filename}')
            print(image_url)
            return render_template('submit.html',pred=pred,confidence=confidence,image_url=image_url,description=desc,prevention=prev)
        else:
            message = 'The given image is not a tomato leaf!!'
            return render_template('index.html', tomato=False, message=message)
        
    return redirect(url_for('index/#disease'))
       
if __name__ == '__main__':
    app.run(debug=True, port=5001)