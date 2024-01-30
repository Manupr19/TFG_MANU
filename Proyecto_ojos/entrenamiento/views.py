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
from datetime import datetime

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
                        shape = predictor(gray, rect)

                        # Ojo Izquierdo
                        xmin_izq = shape.part(36).x - 20
                        xmax_izq = shape.part(39).x + 20
                        ymin_izq = min(shape.part(37).y, shape.part(38).y, shape.part(41).y, shape.part(40).y) - 10
                        ymax_izq = max(shape.part(37).y, shape.part(38).y, shape.part(41).y, shape.part(40).y) + 10

                        # Ojo Derecho
                        xmin_der = shape.part(42).x - 20
                        xmax_der = shape.part(45).x + 20
                        ymin_der = min(shape.part(43).y, shape.part(44).y, shape.part(47).y, shape.part(46).y) - 10
                        ymax_der = max(shape.part(43).y, shape.part(44).y, shape.part(47).y, shape.part(46).y) + 10
                              # Verificación de coordenadas válidas y recorte de ojos
                        if xmin_izq >= 0 and ymin_izq >= 0 and xmax_izq < img.shape[1] and ymax_izq < img.shape[0]:
                            ojo_izquierdo = img[ymin_izq:ymax_izq, xmin_izq:xmax_izq]
                            ruta_ojo_izq = f'{directorio_ojos}ojoizq_{archivo}'
                            cv2.imwrite(ruta_ojo_izq, ojo_izquierdo)

                        if xmin_der >= 0 and ymin_der >= 0 and xmax_der < img.shape[1] and ymax_der < img.shape[0]:
                            ojo_derecho = img[ymin_der:ymax_der, xmin_der:xmax_der]
                            ruta_ojo_der = f'{directorio_ojos}ojoder_{archivo}'
                            cv2.imwrite(ruta_ojo_der, ojo_derecho)
                                                            
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

#vista de entrenador
def entrenador_views(request):
       if not request.user.is_authenticated:
        messages.error(request, "Debes iniciar sesión para acceder al entrenamiento.")
        return redirect('/logeate')
       else:
        limpia()
        return render(request, 'entrenador.html')
       
#funcion que limpia al inicio de cada entrenamiento   
def limpia():
    training_data_derecho = None
    training_labels_derecho = None
    training_data_izquierdo = None
    training_labels_izquierdo = None
    if os.path.exists('knn_derecho.xml'):
        os.remove('knn_derecho.xml')
    if os.path.exists('knn_izquierdo.xml'):
        os.remove('knn_izquierdo.xml')

# crea carpetas de las diferentes imagenes
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
training_data_derecho = None
training_labels_derecho = None
training_data_izquierdo = None
training_labels_izquierdo = None
ultima_prediccion_izq, ultima_prediccion_der = None, None
contador_ojo_izquierdo, contador_ojo_derecho = 0, 0
@csrf_exempt
def entrenar_knn_para_todos_los_ojos(request):
    global training_data_derecho, training_labels_derecho
    global training_data_izquierdo, training_labels_izquierdo

    if request.method == 'POST':
        try:
            panel_numero = int(request.POST['panel_numero'])
            vecinos = 3
            directorio_base = "../Proyecto_ojos/entrenamiento/fotos/fotosnormalizadas/"

            # Acumula los datos de entrenamiento para el ojo derecho
            nuevos_datos_d, nuevas_etiquetas_d = procesar_datos_panel(vecinos, panel_numero, 'ojoder', directorio_base)
            training_data_derecho, training_labels_derecho = agregar_datos_entrenamiento(nuevos_datos_d, nuevas_etiquetas_d, training_data_derecho, training_labels_derecho)
            print("training datad: ",training_data_derecho)
            print("training label: ",training_labels_derecho)
            # Acumula los datos de entrenamiento para el ojo izquierdo
            nuevos_datos_i, nuevas_etiquetas_i = procesar_datos_panel(vecinos, panel_numero, 'ojoizq', directorio_base)
            training_data_izquierdo, training_labels_izquierdo = agregar_datos_entrenamiento(nuevos_datos_i, nuevas_etiquetas_i, training_data_izquierdo, training_labels_izquierdo)
            print("training datai: ",training_data_izquierdo)
            print("training label: ",training_labels_izquierdo)
            if panel_numero==6:
                entrenar_y_guardar_modelos()
            # Responder con el estado actual del proceso
            return JsonResponse({'status': 'Datos acumulados para panel ' + str(panel_numero)})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'status': 'Error en la solicitud'}, status=400)
    
def entrenar_y_guardar_modelos():
    global training_data_derecho, training_labels_derecho
    global training_data_izquierdo, training_labels_izquierdo

    if training_data_derecho is not None:
        knn_derecho = entrenar_knn(training_data_derecho, training_labels_derecho)
        knn_derecho.save('knn_derecho.xml')

    if training_data_izquierdo is not None:
        knn_izquierdo = entrenar_knn(training_data_izquierdo, training_labels_izquierdo)
        knn_izquierdo.save('knn_izquierdo.xml')

def agregar_datos_entrenamiento(nuevos_datos, nuevas_etiquetas, datos_existentes, etiquetas_existentes):
    # Asegúrate de que las nuevas etiquetas sean un vector columna
    nuevas_etiquetas = nuevas_etiquetas.reshape(-1, 1)

    if datos_existentes is None:
        datos_existentes = nuevos_datos
        etiquetas_existentes = nuevas_etiquetas
    else:
        datos_existentes = np.vstack((datos_existentes, nuevos_datos))
        etiquetas_existentes = np.vstack((etiquetas_existentes, nuevas_etiquetas))
    
    return datos_existentes, etiquetas_existentes

def procesar_datos_panel(vecinos, panel_numero, ojo, directorio_base):
    etiqueta = panel_numero
    tamano_imagen = (100, 100)
    trainingdata = None
    trainingLabels = None

    input_images_path = os.path.join(directorio_base, f'panel{panel_numero}/')
    file_pattern = os.path.join(input_images_path, f'{ojo}_panel{panel_numero}_*.jpg')
    print("El path que utiliza:", file_pattern)
    files_names = glob.glob(file_pattern)

    for fichpath in files_names:
        print(f"Intentando cargar la imagen: {fichpath}")
        img = cv2.imread(fichpath)
        if img is None:
            print(f"No se pudo cargar la imagen: {fichpath}")
            continue
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

        print(f"Imagen procesada: {fichpath}, Etiqueta asignada: {etiqueta}")

    return trainingdata, trainingLabels

def entrenar_knn(trainingdata, trainingLabels):
    if trainingdata is not None and trainingLabels is not None:
        # Asegúrate de que trainingLabels sea un vector columna
        trainingLabels = trainingLabels.reshape(-1, 1)

        knn = cv2.ml.KNearest_create()
        print('ANTES DE ENTRENAR')
        print('Training data:', trainingdata)
        print(cv2.ml.ROW_SAMPLE)
        print('Training labels:', trainingLabels)
        knn.train(trainingdata, cv2.ml.ROW_SAMPLE, trainingLabels)
        return knn
    else:
        print('No se han procesado datos de entrenamiento. Comprueba las rutas y los datos de entrada.')
        return None

def todos_vecinos_iguales(vecinos, prediccion):
    for vecino in vecinos:
        if vecino != prediccion:
            return False
    return True

# Función auxiliar para verificar si dos vecinos son iguales
#def dos_vecinos_coinciden(vecinos):
  #  return vecinos[0] == vecinos[1] or vecinos[0] == vecinos[2] or vecinos[1] == vecinos[2]


def todos_vecinos_iguales(vecinos):
    return all(v == vecinos[0] for v in vecinos)


ultima_prediccion = None
contador_coincidencias = 0

# Método actua
@csrf_exempt
def actua(request):
    global ultima_prediccion,contador_coincidencias
    if request.method == 'POST':
         # Inicializa panel_seleccionado al principio        
        panel_seleccionado = None      
        knn_izquierdo_path = os.path.abspath('knn_izquierdo.xml')
        knn_derecho_path = os.path.abspath('knn_derecho.xml')
        
        try:
            knn_izquierdo = cv2.ml.KNearest_load(knn_izquierdo_path)
            knn_derecho = cv2.ml.KNearest_load(knn_derecho_path)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
        
        try:
            cap = cv2.VideoCapture(0)
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
                    xmin_izq = shape[36][0] - 20
                    xmax_izq = shape[39][0] + 20
                    ymin_izq = min(shape[37][1], shape[38][1], shape[41][1], shape[40][1]) - 10
                    ymax_izq = max(shape[37][1], shape[38][1], shape[41][1], shape[40][1]) + 10
                     # Ojo Derecho
                    xmin_der = shape[42][0] - 20
                    xmax_der = shape[45][0] + 20
                    ymin_der = min(shape[43][1], shape[44][1], shape[47][1], shape[46][1]) - 10
                    ymax_der = max(shape[43][1], shape[44][1], shape[47][1], shape[46][1]) + 10

                    # Redimensionamiento (si es necesario para mantener el tamaño consistente)
                    ojo_izquierdo = img[ymin_izq:ymax_izq, xmin_izq:xmax_izq]
                    ojo_derecho = img[ymin_der:ymax_der, xmin_der:xmax_der]

                    # Normalizamos imagen
                    norm_img = np.zeros((70, 40))
                    ojoizqnorm = cv2.normalize(ojo_izquierdo, norm_img, 0, 255, cv2.NORM_MINMAX)
                    ojodernorm = cv2.normalize(ojo_derecho, norm_img, 0, 255, cv2.NORM_MINMAX)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # PREDECIMOS CON LOS KNN

                    # Procesamiento para el ojo izquierdo
                    try:
                        izqgray = cv2.cvtColor(ojoizqnorm, cv2.COLOR_BGR2GRAY)
                        izqecu = cv2.equalizeHist(izqgray)
                        izqecu_resized = cv2.resize(izqecu, (100, 100)) 
                        izqecu1d = np.reshape(izqecu_resized, (1, -1)).astype(np.float32)  # Asegúrate de que sea float32
                        tuplaizq = izqecu1d.astype(np.float32)

                        retizq, resultizq, neighboursizq, distizq = knn_izquierdo.findNearest(tuplaizq, 3)
                        print("Resultado izquierdo:", resultizq, "Vecinos:", neighboursizq)
                    except Exception as e:
                        print("Error al realizar predicción con KNN para el ojo izquierdo:", e)

                    # Procesamiento para el ojo derecho
                    try:
                        dergray = cv2.cvtColor(ojodernorm, cv2.COLOR_BGR2GRAY)
                        derecu = cv2.equalizeHist(dergray)
                        derecu_resized = cv2.resize(derecu, (100, 100))
                        derecu1d = np.reshape(derecu_resized, (1, -1)).astype(np.float32)  # Asegúrate de que sea float32
                        tuplader = derecu1d.astype(np.float32)

                        retder, resultder, neighboursder, distder = knn_derecho.findNearest(tuplader, 3)
                        print("Resultado derecho:", resultder, "Vecinos:", neighboursder)
                    except Exception as e:
                        print("Error al realizar predicción con KNN para el ojo derecho:", e)


                if  todos_vecinos_iguales(neighboursizq[0]) and not todos_vecinos_iguales(neighboursder[0]):
                        panel_seleccionado = int(resultizq)
                        print('Todos los vecinos del ojo izquierdo son iguales')
                    # Verificar si todos los vecinos del ojo derecho son iguales
                if todos_vecinos_iguales(neighboursder[0]) and not todos_vecinos_iguales(neighboursizq[0]):
                        panel_seleccionado = int(resultder)
                        print('Todos los vecinos del ojo derecho son iguales')
                  
                   #POSIBLE MEJORA     
                    # Nueva condición: si dos vecinos de un ojo coinciden y el resultado del otro ojo es uno de esos vecinos
                #if dos_vecinos_coinciden(neighboursizq[0]) and resultder in neighboursizq[0]:
                 #       panel_seleccionado = int(resultizq)
                  #      print('Dos vecinos del ojo izquierdo coinciden y uno del derecho también')
                        

                #if dos_vecinos_coinciden(neighboursder[0]) and resultizq in neighboursder[0]:
                 #       panel_seleccionado = int(resultder)
                  #      print('Dos vecinos del ojo derecho coinciden y uno del izquierdo también')
                
                
                if resultizq == resultder:
                    if ultima_prediccion == resultizq:
                        contador_coincidencias += 1
                        print('incrementamos')
                    else:
                        contador_coincidencias = 1
                        ultima_prediccion = resultizq
                        print('empezamos')

                    if contador_coincidencias >= 4:
                        panel_seleccionado = int(resultizq) 
                        contador_coincidencias = 0  # Restablecer el contador para la próxima vez
                        print('lo hemos conseguido yuju')
                        
             

        except Exception as e:
         return JsonResponse({'error': str(e)}, status=500)
        finally:
            # Liberar la captura de video al salir
            cap.release()

        return JsonResponse({'mensaje': 'Procesamiento completado', 'panelSeleccionado': panel_seleccionado })
    else:
        return JsonResponse({'mensaje': 'Método no permitido'}, status=405)
    
    