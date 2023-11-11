from django.shortcuts import render
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import dlib
import numpy as np
from django.shortcuts import render
import cv2
import glob

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
                                print("No se detectaron caras en la imagen")
                    else:
                        # Manejar el caso en el que no se detectaron caras en la imagen
                        print("No se detectaron caras en la imagen")
                else:
                    # Manejar el caso en el que la imagen no se cargó correctamente
                    print(f"No se pudo cargar la imagen: {img_path}")
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'message': 'Ojos clasificados y guardados correctamente'})



def entrenador_views(request):
    return render(request, 'entrenador.html')

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



# Ruta base donde se encuentran las imágenes de entrenamiento normalizadas
@csrf_exempt
def entrenar_knn_para_todos_los_ojos(request):
    # Verifica si la solicitud es un POST
    if request.method == 'POST':
        try:
            # Extrae datos del formulario POST: número de panel y número de vecinos (fijo en 3)
            panel_numero = int(request.POST['panel_numero'])
            vecinos = 3
      
            # Define la ruta base donde se encuentran las imágenes normalizadas
            directorio_base = "../Proyecto_ojos/entrenamiento/fotos/fotosnormalizadas/"

            # Entrena el modelo KNN para el ojo derecho
            knn_derecho = entrenaoKNN(vecinos, panel_numero, 'der', directorio_base)
           
            # Entrena el modelo KNN para el ojo izquierdo
            knn_izquierdo = entrenaoKNN(vecinos, panel_numero, 'izq', directorio_base)

            # Devuelve una respuesta JSON indicando que el entrenamiento se completó
            return JsonResponse({'status': 'Entrenamiento completado'})
        except Exception as e:
            # Maneja cualquier excepción devolviendo un error en formato JSON
            return JsonResponse({'error': str(e)}, status=500)

    # Devuelve un error si la solicitud no es POST
    return JsonResponse({'status': 'Error en la solicitud'}, status=400)

def entrenaoKNN(vecinos, panel_numero, ojo, directorio_base):
    # Genera nombres de clases basados en el ojo y el número de panel
    clases = [f'ojo{ojo}_panel{panel_numero}_foto{i}' for i in range(1, 4)]
    etiqueta = 0
    trainingdata = None
    trainingLabels = None

    # Define un tamaño estándar para todas las imágenes
    tamano_imagen = (50, 50)

    for clase in clases:
        # Genera el nombre base para las imágenes de la clase actual
        image_base_name = f'{clase}.jpg'
        # Construye la ruta de entrada para las imágenes
        input_images_path = os.path.join(directorio_base, f'panel{panel_numero}/')
        # Establece un patrón para buscar archivos JPEG
        file_pattern = os.path.join(input_images_path, '*.jpg')
        # Obtiene los nombres de los archivos que coinciden con el patrón
        files_names = glob.glob(file_pattern)

        # Imprime la ruta y los nombres de los archivos para verificación
        print(f'Ruta de la carpeta de imágenes: {input_images_path}')
        print(f'El file_pattern es: {file_pattern}')
        print(f'El files_names es: {files_names}')

        for fichpath in files_names:
            # Lee, redimensiona y convierte la imagen a escala de grises y luego la ecualiza
            img = cv2.imread(fichpath)
            img = cv2.resize(img, tamano_imagen)
            imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgecu = cv2.equalizeHist(imggray)
            imgecu1d = np.reshape(imgecu, (1, -1))

            # Prepara los datos de entrenamiento y las etiquetas
            if trainingdata is None:
                trainingdata = imgecu1d.astype(np.float32)
                trainingLabels = np.array([[etiqueta]]).astype(np.float32)
            else:
                trainingdata = np.append(trainingdata, imgecu1d.astype(np.float32), 0)
                trainingLabels = np.append(trainingLabels, np.array([[etiqueta]]).astype(np.float32), 0)

            etiqueta += 1

        # Imprime la forma de los datos de entrenamiento para verificación
        if trainingdata is not None:
            print(trainingdata.shape)

    # Crea y entrena el modelo KNN
    knn = cv2.ml.KNearest_create()
    knn.train(trainingdata, cv2.ml.ROW_SAMPLE, trainingLabels)

    # Encuentra el vecino más cercano y verifica la precisión
    ret, result, neighbours, dist = knn.findNearest(trainingdata, k=vecinos)
    if trainingLabels is not None:
        correct = np.count_nonzero(result == trainingLabels)
        print(f'Aciertos ojo{ojo} = {correct}')
    else:
        print('No hay etiquetas de entrenamiento disponibles.')

    # Devuelve el modelo KNN entrenado
    return knn
