# super_resolution_app/views.py

from django.shortcuts import render
from .models import UploadedImage
from django.conf import settings
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

def index(request):
    if request.method == 'POST' and request.FILES['image']:
        image_file = request.FILES['image']
        uploaded_image = UploadedImage.objects.create(image=image_file)
        # Load the trained autoencoder model
        model_path = os.path.join(settings.BASE_DIR, 'autoencoder.h5')
        model = load_model(model_path)
        # Perform super-resolution on the uploaded image
        uploaded_image_path = os.path.join(settings.MEDIA_ROOT, uploaded_image.image.name)
        uploaded_image_data = np.array(Image.open(uploaded_image_path))
        super_resolved_image_data = model.predict(np.expand_dims(uploaded_image_data, axis=0))[0]
        # Save the super-resolved image
        super_resolved_image_path = os.path.join(settings.MEDIA_ROOT, 'super_resolved_image.jpg')
        Image.fromarray(super_resolved_image_data).save(super_resolved_image_path)

        # Return the path to the super-resolved image to display in the template
        return render(request, 'index.html', {'super_resolved_image': super_resolved_image_path})
    return render(request, 'index.html')
