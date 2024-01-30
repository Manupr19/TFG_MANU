from django.shortcuts import render
from django.shortcuts import render
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.shortcuts import render, redirect
from .forms import UserRegistrationForm
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth import logout
from django.shortcuts import redirect


def panel_view(request):
      if not request.user.is_authenticated:
        messages.error(request, "Debes iniciar sesión para acceder al entrenamiento.")
        return redirect('/logeate')
      else:
            return render(request, 'panel.html')

def inicio_view(request):
    return render(request, 'Inicio.html')

def login_view(request):
    return render(request, 'login.html')

def registro_view(request):
    return render(request, 'registro.html')

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            # Crea un nuevo usuario, pero aún no lo guardes en la base de datos
            new_user = form.save(commit=False)
            # Establece la contraseña elegida
            new_user.set_password(form.cleaned_data['password'])
            # Guarda el objeto User
            new_user.save()
            return redirect('/login')  # Redirige a la página de inicio de sesión
    else:
        form = UserRegistrationForm()

    return render(request, 'registro.html', {'form': form})

def seleccion_panel(request, panel_numero):
    nombre_plantilla = f"seleccionpanel{panel_numero}.html"
    return render(request, nombre_plantilla)

def logueate(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, f'Bienvenido/a, {user.username}!')
            # Redirige a la página de inicio o a la que prefieras después del inicio de sesión
            return redirect('/inicio')
        else:
            # Enviar un mensaje de error si la autenticación falla
            messages.error(request, 'Nombre de usuario o contraseña incorrectos')

    return render(request, 'login.html')  # Asegúrate de que este es el nombre correcto de tu plantilla HTML

def deslog(request):
    logout(request)
    # Puedes redirigir a la página de inicio o a donde prefieras después de cerrar la sesión
    return redirect('/inicio')