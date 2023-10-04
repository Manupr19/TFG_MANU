from django.shortcuts import render
import os
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Create your views here.
directorio_base = '.' + os.sep + 'Entrena6' + os.sep
def entrenador_views(request):
    return render(request, 'entrenador.html')

def crea_directorios():
    try:
        os.stat(directorio_base)
    except:
        os.mkdir(directorio_base)

    dirojos = ['ojoder', 'ojoizq']
    for dir in dirojos:
        try:
            os.stat(directorio_base + dir + os.sep)
        except:
            os.mkdir(directorio_base + dir + os.sep)
        for etiq in range(9):
            try:
                os.stat(directorio_base + dir + os.sep + dir + str(etiq))
            except:
                os.mkdir(directorio_base + dir + os.sep + dir + str(etiq))
def crear_directorios_view(request):
    crea_directorios()
    return HttpResponse("Directorios creados con éxito")

@csrf_exempt
def guardar_imagenes(request):
    if request.method == 'POST':
        try:
            # Asegúrate de que la carpeta "fotos" existe en la aplicación "entrenamiento"
            fotos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fotos')
            if not os.path.exists(fotos_dir):
                os.makedirs(fotos_dir)

            # Guarda las imágenes en la carpeta "fotos"
            for key, file in request.FILES.items():
                with open(os.path.join(fotos_dir, file.name), 'wb') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)

            return JsonResponse({'message': 'Imágenes guardadas correctamente.'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Método no permitido'}, status=405)