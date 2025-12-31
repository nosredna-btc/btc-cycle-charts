#!/bin/bash
python3 allcharts.py && mv charts_dev/* charts && python3 -m http.server 8000
