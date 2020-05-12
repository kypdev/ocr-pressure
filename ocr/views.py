from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import os
import json


def ocr(request):
    def sys():
        pressure = {
        "sys": "126",
        "dia": "91",
        "p": "66"
        }
        return pressure
    return JsonResponse(sys(), safe=False)
    # return HttpResponse(json.dumps(pressure))


def req_ocr(request):
    # os.system("echo wow")
    # os.system("echo %CD%")
    os.system("cd ocr_scripts && dir && python crop.py")
    os.system("cd ocr_scripts && dir && python ocr_sys.py")
    with open('ocr/data.json') as f:
        data = json.load(f)
        print(data)
    os.system("cd ocr && dir && del data.json")

    return JsonResponse(data, safe=False)