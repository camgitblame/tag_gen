#!/bin/bash

# Test script for the FastAPI backend
echo "Testing FastAPI backend..."

BASE_URL="http://localhost:8080"

echo "1. Testing health check..."
curl -s "$BASE_URL/" | python -m json.tool

echo -e "\n2. Testing movies endpoint..."
curl -s "$BASE_URL/movies" | head -c 200
echo "..."

echo -e "\n3. Testing movie details endpoint..."
curl -s "$BASE_URL/movie/Toy%20Story" | python -m json.tool

echo -e "\nTests complete!"
