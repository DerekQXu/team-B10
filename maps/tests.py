from django.test import TestCase
from django.http import HttpRequest
from django.test import SimpleTestCase
from django.urls import reverse

from . import views
from . import placesapi

class PageTests(SimpleTestCase):

    def home_page_status_code(self):
        response = self.client.get('/')
        self.assertEquals(response.status_code, 200)

    def maps_view_uses_correct_template(self):
        response = self.client.get('maps/')
        self.assertEquals(response.status_code, 200)
        self.assertTemplateUsed(response, 'maps.html')

    def about_view_uses_correct_template(self):
        response = self.client.get('about/')
        self.assertEquals(response.status_code, 200)
        self.assertTemplateUsed(response, 'about.html')

class PlacesTests(TestCase):

    def test_places_search(self):
        api = "AIzaSyCn_fbVwGYtPjSfCTZDv97SrLpMOrGmMt8"
        expected_result = [{'geometry': {'location': {'lat': 34.06920639999999, 'lng': -118.4429419}, 'viewport': {'northeast': {'lat': 34.07053193029149, 'lng': -118.4418683197085}, 'southwest': {'lat': 34.06783396970849, 'lng': -118.4445662802915}}}, 'icon': 'https://maps.gstatic.com/mapfiles/place_api/icons/school-71.png', 'id': '99aad41e29fcfeaf94a8f1daa7701cdd36c8e3e2', 'name': 'UCLA Planetarium', 'photos': [{'height': 3024, 'html_attributions': ['<a href="https://maps.google.com/maps/contrib/107191420925388675995">jon chua</a>'], 'photo_reference': 'CmRaAAAAX3cXuqAW0SevSK81Fv1tAZO20L-08yRpyMnmLwuMlReni19SpY1WSOqQD5ezCIBh3nACRF3VTd5vXga0g1JXB9swFe5HuyRWFHrrcJntARiNZIj27jJB0SEsRpN0vY1cEhCR7P8ak21IesQrCDD6OtzCGhQYSMMcKTYc1Qdo7LmNJ-QYuUvqiw', 'width': 4032}], 'place_id': 'ChIJVVWVhIi8woARraSRk6QAI5w', 'plus_code': {'compound_code': '3H94+MR Los Angeles, California, United States', 'global_code': '85633H94+MR'}, 'rating': 5, 'reference': 'ChIJVVWVhIi8woARraSRk6QAI5w', 'scope': 'GOOGLE', 'types': ['tourist_attraction', 'point_of_interest', 'establishment'], 'user_ratings_total': 2, 'vicinity': '405 Hil Gard Avenue #8224, Los Angeles'}]
        places = placesapi.search_places("34.069206,-118.4429419", "30", "tourist_attraction", api)
        self.assertEquals(places[0]['geometry'], expected_result[0]['geometry'])
