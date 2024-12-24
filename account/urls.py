from django.urls import path
from . import views

urlpatterns = [

    path('login/', views.signIn, name='login'),
    path('sign-up/', views.signUp, name='signUp'),
    path('logout/', views.lgOut, name='logout'),
]
