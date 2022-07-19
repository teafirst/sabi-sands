from django.shortcuts import render
from django.http import HttpResponse
import tensorflow as tf

from .forms import ImageForm

# Create your views here.
def hello(request):
    if request.method=='POST':
        
        # Database
        ## Do WE NEED TO SAVE IT TO DATABASE?
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            img_obj = form.instance
            apply_something(img_obj)
            return render(request, 'homeApp/index.html', {'image':img_obj})
        
    form = ImageForm()
    return render(request, 'homeApp/index.html', {
        'form':form
    })

def apply_something(img_obj):
    # print(img_obj.first_image.url)
    max_dim = 512
    img = tf.io.read_file(img_obj.first_image.url)

    ## Converts the string into a tensor
    img = tf.image.decode_image(img, channels=3)

    ## Converts image inputs to float
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape*scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    print(img)