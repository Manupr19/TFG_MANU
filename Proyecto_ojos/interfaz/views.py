from django.shortcuts import render
from django.shortcuts import render
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login, authenticate
from .forms import CustomUserCreationForm
# Create your views here.

def panel_view(request):
    
    return render(request, 'panel.html')

def inicio_view(request):
    return render(request, 'Inicio.html')

def login_view(request):
    return render(request, 'login.html')
def registro_view(request):
    return render(request, 'registro.html')



def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('/inicio')  # Redirige a la página de inicio después del registro
    else:
        form = CustomUserCreationForm()
    return render(request, 'register.html', {'form': form})