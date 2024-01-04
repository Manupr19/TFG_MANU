import os
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import dlib
import numpy as np
from django.shortcuts import render
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import glob
from django.shortcuts import redirect
from django.contrib import messages
import time


# Inicializa un arreglo para rastrear el estado de los botones
pulsado = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# Modelo de detección de caras
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')
vecinos=3
direc_base='/entrenamientoS'

# Método de vista para clasificar ojos
@csrf_exempt
def clasificar(request):
    global directorio_base

    # Obtener el número de panel desde la solicitud
    numero_seleccionado = request.POST['panel_numero']
    # Directorio de imágenes capturadas desde el primer código
    directorio_imagenes = f'entrenamiento/fotos/panel{numero_seleccionado}/'
    # Directorio donde se guardarán los ojos normalizados
    directorio_ojos = f'entrenamiento/fotos/fotosnormalizadas/panel{numero_seleccionado}/'
    try:
        # Verificar si el directorio de ojos normalizados existe, si no, créalo
        if not os.path.exists(directorio_ojos):
            os.makedirs(directorio_ojos)
        # Listar archivos en el directorio de imágenes capturadas
            archivos = os.listdir(directorio_imagenes)
        for archivo in archivos:
            if archivo.endswith('.jpg'):
                img_path = os.path.join(directorio_imagenes, archivo)
                # Leer la imagen desde el archivo
                img = cv2.imread(img_path)
                # Verificar si la imagen se cargó correctamente
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Detectar caras en la imagen
                    rects = detector(gray)
                    if len(rects) > 0:
                        for rect in rects:
                            # Obtener landmarks faciales
                            shape = predictor(gray, rect)
                            # Verificar que se detectaron caras
                            if len(shape.parts()) >= 68:
                                # Recortar y guardar los ojos izquierdo y derecho
                         # Ajusto los puntos de inicio y fin para una región mucho más grande
                                x1_left, y1_left = shape.part(36).x - 20, shape.part(37).y - 20
                                x2_left, y2_left = shape.part(39).x + 20, shape.part(41).y + 20
                                x1_right, y1_right = shape.part(42).x - 20, shape.part(43).y - 20
                                x2_right, y2_right = shape.part(45).x + 20, shape.part(47).y + 20
                                # Verifica que las coordenadas sean válidas
                                if x1_left >= 0 and y1_left >= 0 and x2_left >= 0 and y2_left >= 0:
                                    # Realiza el recorte y guarda el ojo izquierdo
                                    ojo_recortado1 = img[y1_left:y2_left, x1_left:x2_left]
                                    ruta_ojo_izq = f'{directorio_ojos}ojoizq_{archivo}'
                                    cv2.imwrite(ruta_ojo_izq, ojo_recortado1)

                                if x1_right >= 0 and y1_right >= 0 and x2_right >= 0 and y2_right >= 0:
                                    # Realiza el recorte y guarda el ojo derecho
                                    ojo_recortado2 = img[y1_right:y2_right, x1_right:x2_right]
                                    ruta_ojo_der = f'{directorio_ojos}ojoder_{archivo}'
                                    cv2.imwrite(ruta_ojo_der, ojo_recortado2)

                                    
                            else:
                                # Manejar el caso en el que no se detectaron caras en la imagen
                                return JsonResponse({'error': 'No se detectaron caras en la imagen'}, status=400)
                    else:
                        # Manejar el caso en el que no se detectaron caras en la imagen
                        return JsonResponse({'error': 'No se detectaron caras en la imagen'}, status=400)
                else:
                    # Manejar el caso en el que la imagen no se cargó correctamente
                    print(f"No se pudo cargar la imagen: {img_path}")
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'message': 'Ojos clasificados y guardados correctamente'})

def arrayTOimgtk(lectura):
    b, g, r = cv2.split(lectura)
    img = cv2.merge((r, g, b))
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)
    return imgtk

def entrenador_views(request):
       if not request.user.is_authenticated:
        messages.error(request, "Debes iniciar sesión para acceder al entrenamiento.")
        return redirect('/logeate')
       else:
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
            # Accede al parámetro "panel_numero" desde el cuerpo de la solicitud POST
            panel_numero = request.POST.get('panel_numero')

            # Asegúrate de que la carpeta "fotos" existe en la aplicación "entrenamiento"
            fotos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fotos')

            # Crea una subcarpeta dentro de "fotos" con el nombre "panel+numero"
            panel_dir = os.path.join(fotos_dir, f'panel{panel_numero}')
            if not os.path.exists(panel_dir):
                os.makedirs(panel_dir)

            # Guarda las imágenes en la subcarpeta correspondiente
            for key, file in request.FILES.items():
                with open(os.path.join(panel_dir, file.name), 'wb') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)

            return JsonResponse({'message': f'Imágenes guardadas correctamente en la carpeta panel{panel_numero}'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Método no permitido'}, status=405)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# Ruta base donde se encuentran las imágenes de entrenamiento normalizadas


@csrf_exempt
def entrenar_knn_para_todos_los_ojos(request):
    
    if request.method == 'POST':
        try:
            # Obtener datos del formulario POST
            panel_numero = int(request.POST['panel_numero'])
            vecinos = 3
      
            # Definir la ruta base
            directorio_base = "../Proyecto_ojos/entrenamiento/fotos/fotosnormalizadas/"
            # Entrenar KNN para el ojo derecho
            knn_derecho = entrenaoKNN(vecinos, panel_numero, 'der', directorio_base)
            knn_derecho.save('knn_derecho.xml')      
            # Entrenar KNN para el ojo izquierdo
            knn_izquierdo = entrenaoKNN(vecinos, panel_numero, 'izq', directorio_base)
            knn_izquierdo.save('knn_izquierdo.xml')
            #os.chmod('knn_izquierdo.xml', 0o777)

            

            return JsonResponse({'status': 'Entrenamiento completado'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'status': 'Error en la solicitud'}, status=400)

def entrenaoKNN(vecinos, panel_numero, ojo, directorio_base):
    clases = [f'ojo{ojo}_panel{panel_numero}_foto{i}' for i in range(6)]
    etiqueta = 1
    trainingdata = None
    trainingLabels = None
    tamano_imagen = (50, 50)

    for clase in clases:
        input_images_path = os.path.join(directorio_base, f'panel{panel_numero}/')
        file_pattern = os.path.join(input_images_path, '*.jpg')
        files_names = glob.glob(file_pattern)
       # print('len(clases)',len(clases))
        #print('etiqueta:',etiqueta)
        for fichpath in files_names:
            img = cv2.imread(fichpath)
            img = cv2.resize(img, tamano_imagen)
            imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgecu = cv2.equalizeHist(imggray)
            imgecu1d = np.reshape(imgecu, (1, -1))

            if trainingdata is None:
                trainingdata = imgecu1d.astype(np.float32)
                trainingLabels = np.array([[etiqueta]]).astype(np.float32)
            else:
                trainingdata = np.append(trainingdata, imgecu1d.astype(np.float32), 0)
                trainingLabels = np.append(trainingLabels, np.array([[etiqueta]]).astype(np.float32), 0)
                
        # Incrementar la etiqueta después de procesar todas las imágenes de una clase
        etiqueta += 1

    if trainingdata is not None:
        #print('trainingshape:', trainingdata.shape)

        #print(trainingLabels)
        knn = cv2.ml.KNearest_create()
        knn.train(trainingdata, cv2.ml.ROW_SAMPLE, trainingLabels)
        ret, result, neighbours, dist = knn.findNearest(trainingdata, k=3)
        print('resultado prediccion:',result)

    if trainingLabels is not None:
        correct = np.count_nonzero(result == trainingLabels)
        print(f'Aciertos ojo{ojo} = {correct}')
    else:
        print('No hay etiquetas de entrenamiento disponibles.')

    return knn

# Método actua
@csrf_exempt
def actua(request):
    if request.method == 'POST':
        # Captura de video (puedes modificar esta parte según tu necesidad)
        # Cargar modelos KNN
        knn_izquierdo = cv2.ml.KNearest_load('Proyecto_ojos/knn_izquierdo.xml')
        knn_derecho = cv2.ml.KNearest_load('Proyecto_ojos/knn_derecho.xml')

        cap = cv2.VideoCapture(0)
        
        try:
            # Capturar un fotograma
            ret, img = cap.read()

            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detectar caras en el fotograma
            rects = detector(gray, 1)

            if len(rects) > 0:
                # Si hay caras detectadas, realizar el procesamiento aquí
                for rect in rects:
                    shape = predictor(gray, rect)
                    shape = shape_to_np(shape)
                    # Estraigo coordenadas ojo izquierdo
                    xmin = shape[37][0]
                    xmax = shape[40][0]
                    ymax = max(shape[38][1], shape[39][1], shape[41][1], shape[42][1])
                    ymin = min(shape[38][1], shape[39][1], shape[41][1], shape[42][1])

                    # Creamos imagenes del ojo izquierdo
                    ojoizq = img[ymin - 10:ymax + 10, xmin - 20:xmax + 20]
                    ojoizq = cv2.resize(ojoizq, (70, 40), interpolation=cv2.INTER_AREA)

                    # Estraigo coordenadas ojo derecho
                    xmin = shape[42][0]
                    xmax = shape[45][0]
                    ymax = max(shape[43][1], shape[44][1], shape[46][1], shape[47][1])
                    ymin = min(shape[43][1], shape[44][1], shape[46][1], shape[47][1])

                    # Creamos imagenes del ojo derecho
                    ojoder = img[ymin - 10:ymax + 10, xmin - 20:xmax + 20]
                    ojoder = cv2.resize(ojoder, (70, 40), interpolation=cv2.INTER_AREA)

                    # Normalizamos imagen
                    norm_img = np.zeros((70, 40))
                    ojoizqnorm = cv2.normalize(ojoizq, norm_img, 0, 255, cv2.NORM_MINMAX)
                    ojodernorm = cv2.normalize(ojoder, norm_img, 0, 255, cv2.NORM_MINMAX)
                    
                    # PREDECIMOS CON LOS KNN
                    # Ecualizamos histograma izquierdo
                    izqgray = cv2.cvtColor(ojoizqnorm, cv2.COLOR_BGR2GRAY)
                    izqecu = cv2.equalizeHist(izqgray)
                    izqecu1d = np.reshape(izqecu, (1, 2800))
                    tuplaizq = izqecu1d.astype(np.float32)
                    retizq, resultizq, neighboursizq, distizq = knn_izquierdo.findNearest(tuplaizq, 3)
                    print(resultizq, neighboursizq, distizq)

                    # Ecualizamos histograma derecho
                    dergray = cv2.cvtColor(ojodernorm, cv2.COLOR_BGR2GRAY)
                    derecu = cv2.equalizeHist(dergray)
                    derecu1d = np.reshape(derecu, (1, 2800))
                    tuplader = derecu1d.astype(np.float32)
                    retder, resultder, neighboursder, distder = knn_derecho.findNearest(tuplader, 3)
                    print(resultder, neighboursder, distder)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        finally:
            # Liberar la captura de video al salir
            cap.release()

        return JsonResponse({'mensaje': 'Procesamiento completado'})
    else:
        return JsonResponse({'mensaje': 'Método no permitido'}, status=405)