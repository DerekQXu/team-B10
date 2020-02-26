from django.shortcuts import render

def maps(request):
    return render(request, 'maps/maps.html')

def about(request):
    return render(request, 'maps/about.html', {'title': 'About'})
