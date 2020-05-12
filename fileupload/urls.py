from django.conf.urls import url, include
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from ocr import views

urlpatterns = [
  url(r'^admin/', admin.site.urls),
  url(r'^file/', include('file_app.urls')),
  url(r'^ocr/', views.ocr, name='ocr'),
  url(r'^req-ocr/', views.req_ocr, name='req_ocr')
]

if settings.DEBUG:
  urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)