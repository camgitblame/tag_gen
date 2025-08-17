#!/bin/bash

# Development setup script
echo "Setting up development environment..."

# Backend setup
echo "Setting up FastAPI backend..."
cd backend
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo "Backend setup complete! To run the backend:"
echo "cd backend && source venv/bin/activate && uvicorn main:app --reload --port 8080"

# Frontend setup
echo "Setting up Next.js frontend..."
cd ../frontend
npm install

echo "Frontend setup complete! To run the frontend:"
echo "cd frontend && npm run dev"

echo ""
echo "=== Development URLs ==="
echo "Backend API: http://localhost:8080"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8080/docs"
