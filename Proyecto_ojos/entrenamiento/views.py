from django.shortcuts import render

# Create your views here.
def entrenador_views(request):
    return render(request, 'entrenador.html')