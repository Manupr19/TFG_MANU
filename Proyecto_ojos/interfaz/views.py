from django.shortcuts import render

# Create your views here.

def panel_view(request):
    letras = ["A", "B", "C", "D", "E", "F"]
    return render(request, 'Proyecto_ojos\interfaz\panel.html', {'letras': letras})
