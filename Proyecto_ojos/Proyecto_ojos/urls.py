from django.contrib import admin
from django.urls import path
from interfaz.views import panel_view as panel
from interfaz.views import login_view as login
from interfaz.views import inicio_view as inicio
from entrenamiento.views import entrenador_views as entrenador
from entrenamiento.views import guardar_imagenes as guardar
from entrenamiento.views import tomar_fotos as tomar
from entrenamiento.views import entrena as entrena
urlpatterns = [
    
    path('admin/', admin.site.urls),
    path('panel/', panel, name='panel_view'),
    path('login/', login, name='login_view'),
    path('tomarfotos/', tomar, name='tomarfotos'),
    path('entrena/', entrena, name='entrena'),
    
    path('entrenamiento/', entrenador, name='entrenador_view'),
    path('', inicio, name='inicio_view'),
    path('inicio/', inicio, name='inicio_view'),
    path('guardar_imagenes/', guardar, name='guardar_imagenes'),


]
