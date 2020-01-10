from django.conf.urls import url
from . import views

app_name = 'identify'

urlpatterns = [
    url(r'^$', views.home, name='home'),
    url(r'^index$', views.index, name='index'),
    url(r'^search$', views.search, name='search'),
    url(r'^history$', views.history, name='history'),
]