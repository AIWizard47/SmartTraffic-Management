from django.shortcuts import render

# Create your views here.
import uuid
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from .models import Profile
from django.http import HttpResponseRedirect, HttpResponse


# Create your views here.

def signIn(request):
    error_mess = None
    if request.method == "POST":
        email = request.POST.get("username").strip()
        password = request.POST.get("password").strip()

        user_obj = User.objects.filter(username=email)
        if not user_obj.exists():
            error_mess = 'Account Not found'
        # elif not user_obj[0].profile.is_email_verified:
        #     error_mess = 'Your account is not verified'
        else:
            user_obj = authenticate(username=email, password=password)
            if user_obj:
                login(request, user_obj)
                return HttpResponseRedirect("/")
            else:
                error_mess = 'Invalid credentials'

    context = {
        'title': 'Login',
        'error_mess': error_mess
    }
    return render(request, 'login.html', context=context)


def signUp(request):
    error_mess = None
    if request.method == "POST":
        first_name = request.POST.get("name").strip()
        email = request.POST.get("email").strip()
        number = request.POST.get("number").strip()
        password = request.POST.get("password").strip()
        confirm_password = request.POST.get("confirm_password").strip()

        if User.objects.filter(username=email).exists():
            error_mess = 'Email already exists'
        elif len(number) != 10:
            error_mess = 'Invalid number'
        elif len(password) < 8:
            error_mess = 'Passwords must be more then 8 letter'
        elif password != confirm_password:
            error_mess = 'Passwords do not match'
        else:
            user_obj = User.objects.create(
                first_name=first_name,
                username=email,
                email=email
            )
            user_obj.set_password(password)
            user_obj.save()

            # this use for creating the profile for user by adding extra things like token phone , number etc..
            email_token = str(uuid.uuid4())
            user = Profile.objects.create(user=user_obj, email_token=email_token, phone_number=number)
            # user.is_email_verified = True
            user.save()

            messages.success(request, 'Email has been sent to your email address for verification')
            login(request, user_obj)
            return HttpResponseRedirect("/")
    context = {
        'title': 'Sign In',
        'error_mess': error_mess
    }
    return render(request, 'signup.html', context=context)


def lgOut(request):
    logout(request)
    return HttpResponseRedirect("/")