from django.shortcuts import render


def home(request):
    return render(request, 'maps/home.html')


def survey(request):
    return render(request, 'maps/survey.html')


def about(request):
    return render(request, 'maps/about.html', {'title': 'About'})
