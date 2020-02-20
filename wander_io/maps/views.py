from django.shortcuts import render


def home(request):
    return render(request, 'maps/home.html')


def about(request):
    return render(request, 'maps/about.html', {'title': 'About'})
