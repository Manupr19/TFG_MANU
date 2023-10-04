"""Proyecto_ojos URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from interfaz.views import panel_view as panel
from entrenamiento.views import entrenador_views as entrenador
from entrenamiento.views import guardar_imagenes as guardar
urlpatterns = [
    
    path('admin/', admin.site.urls),
    path('panel/', panel, name='panel_view'),
    path('entrenamiento/', entrenador, name='entrenador_view'),
    path('guardar_imagenes/', guardar, name='guardar_imagenes'),


]
