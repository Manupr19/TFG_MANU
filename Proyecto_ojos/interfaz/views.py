from django.shortcuts import render

# Create your views here.

def panel_view(request):
    
    return render(request, 'panel.html')

def inicio_view(request):
    return render(request, 'Inicio.html')

def login_view(request):
    return render(request, 'login.html')