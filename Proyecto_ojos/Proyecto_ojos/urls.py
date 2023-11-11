from django.contrib import admin
from django.urls import path
from interfaz.views import panel_view as panel
from interfaz.views import login_view as login
from interfaz.views import inicio_view as inicio
from interfaz.views import registro_view as regis
from entrenamiento.views import entrenador_views as entrenador
from entrenamiento.views import guardar_imagenes as guardar
from entrenamiento.views import entrenar_knn_para_todos_los_ojos as entrenadorparatodos
from entrenamiento.views import clasificar as clasifica

urlpatterns = [
    
    path('admin/', admin.site.urls),
    path('panel/', panel, name='panel_view'),
    path('registro/', regis, name='registro_view'),
    path('login/', login, name='login_view'),
    path('clasifica/', clasifica, name='clasifica'),
    path('entrenamiento/', entrenador, name='entrenador_view'),
    path('entrena/', entrenadorparatodos, name='entrenador_view'),
    path('', inicio, name='inicio_view'),
    path('inicio/', inicio, name='inicio_view'),
    path('guardar_imagenes/', guardar, name='guardar_imagenes'),


]
