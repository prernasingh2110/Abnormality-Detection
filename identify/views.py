from django.shortcuts import render
from django.http import HttpResponse
from .models import Xray
from django.template import loader
from .forms import boneform
from keras.preprocessing.image import image
import numpy as np
from .cnn import finger, elbow, humerus, wrist, hand, forearm, shoulder, body_part_classifier



# Create your views here.

def predict(request):
    ans = ['elbow', 'forearm', 'finger', 'hand', 'humerus', 'shoulder', 'wrist']
    need = Xray.objects.last()
    predict_path = "media/" + str(need.bone_image)
    test_image = image.load_img(predict_path, target_size=(224,224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = body_part_classifier.predict(test_image)
    ans2 = 0
    result = result[0]
    result = result.tolist()
    i = 0
    for x in result:
        if x == 1:
            ans2 = ans[i]
        i = i + 1
    need.body_part = ans2
    body_part = ans2
    if body_part == 'elbow':
        condition_classifier = elbow
    elif body_part == 'forearm':
        condition_classifier = forearm
    elif body_part == 'finger':
        condition_classifier = finger
    elif body_part == 'hand':
        condition_classifier = hand
    elif body_part == 'humerus':
        condition_classifier = humerus
    elif body_part == 'wrist':
        condition_classifier = wrist
    elif body_part == 'shoulder':
        condition_classifier = shoulder
    result2 = condition_classifier.predict(test_image)
    result2 = result2[0]
    result2 = result2.tolist()
    condition = 'positive'
    prob = result2[0]
    if result2[1] > result2[0]:
        condition = 'negative'
        prob = result2[1]
    need.condition = condition
    p = prob
    print(str(prob))
    prob = prob*100
    prob = str(round(prob, 2))+"%"
    need.probability = prob
    need.save()
    if p > 0.7:
        prob = "Consultation strongly Advised! High Risk!"
    elif p > .5:
        prob = "Consultation advised, low risk involved."
    else:
        prob = "Exercise advised."
    context = {
        'ans2': ans2,
        'condition':condition,
        'need': need,
        'prob': prob,
    }
    return context



def home(request):
    template = loader.get_template('identify/home.html')
    return HttpResponse(template.render())

def index(request):
    if request.method == 'POST':
        form = boneform(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            context = predict(request)
            template = loader.get_template('identify/prediction.html')
            return HttpResponse(template.render(context, request))
    else:
        form = boneform()
        return render(request, 'identify/index.html', {'form': form})


def search(request):
    fil = request.GET['q']
    images = Xray.objects.filter(patient=fil)
    probs = []
    for i in images:
        p = i.probability[:1]
        if p == '':
        	p = '0'
        p = int(p)
        #print("PROASDASDSAFDSAFDSFASDFSADF  " + p)
        if p > 7:
            prob = "Consultation strongly Advised! High Risk!"
        elif p > 5:
            prob = "Consultation advised, low risk involved."
        else:
            prob = "Exercise advised."
        probs.append(prob)
    context = {
        'all_images': images,
        'probs': probs,
    }
    template = loader.get_template('identify/search.html')
    return HttpResponse(template.render(context, request))


def history(request):
    images = Xray.objects.all()
    probs = []
    for i in images:
        p = i.probability[:1]
        if p == '':
            p = '0'
        p = int(p)
        if p > 7:
            prob = "Consultation strongly Advised! High Risk!"
        elif p > 5:
            prob = "Consultation advised, low risk involved."
        else:
            prob = "Exercise advised."
        probs.append(prob)
    context = {
        'all_images': images,
        'probs': probs,
    }
    template = loader.get_template('identify/search.html')
    return HttpResponse(template.render(context, request))

