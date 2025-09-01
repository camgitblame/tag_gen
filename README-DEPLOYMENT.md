# Tagline Generator - FastAPI + Next.js

A modern web application that generates movie taglines using various AI models. The original Streamlit app has been converted to a FastAPI backend (deployed on Google Cloud Run) with a Next.js frontend (deployed on Vercel).

## Architecture

- **Backend**: FastAPI (Python) - Serves movie data and generated taglines via REST API
- **Frontend**: Next.js (TypeScript + React)
- **Deployment**: 
  - Backend: Google Cloud Run
  - Frontend: Vercel

## Quick Start (Local Development)

1. **Setup development environment:**
   ```bash
   chmod +x setup-dev.sh
   ./setup-dev.sh
   ```

2. **Run backend:**
   ```bash
   cd backend
   source venv/bin/activate
   uvicorn main:app --reload --port 8080
   ```

3. **Run frontend:**
   ```bash
   cd frontend
   npm run dev
   ```

4. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8080
   - API Documentation: http://localhost:8080/docs

## Deployment

### Backend (Google Cloud Run)

1. **Prerequisites:**
   - Google Cloud SDK installed
   - Project with Cloud Run API enabled
   - Docker installed

2. **Deploy:**
   ```bash
   cd backend
   
   # Update PROJECT_ID in deploy.sh
   vim deploy.sh
   
   # Make executable and deploy
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. **Note the deployment URL** - you'll need it for the frontend configuration.

### Frontend (Vercel)

1. **Prerequisites:**
   - Vercel account
   - GitHub repository 

2. **Deploy via Vercel CLI:**
   ```bash
   cd frontend
   npm i -g vercel
   vercel
   ```

3. **Or deploy via GitHub:**
   - Push code to GitHub
   - Connect repository to Vercel
   - Set environment variable: `NEXT_PUBLIC_API_URL=YOUR_CLOUD_RUN_URL`

## Environment Variables

### Frontend
- `NEXT_PUBLIC_API_URL`: Your FastAPI backend URL (Cloud Run URL)

### Backend
- No additional environment variables required for basic setup
- All data files are included in the Docker image

## API Endpoints

- `GET /` - Health check
- `GET /movies` - List all available movies
- `GET /movie/{title}` - Get movie details and generated taglines
- `GET /docs` - FastAPI auto-generated documentation

## Key Features

1. **Movie Selection**: Choose from 1000+ movies
2. **Multiple AI Models**: Compare taglines from 6 different models:
   - Baseline Model
   - RAG At Inference
   - Overview + Genre Model
   - Overview + Genre + RAG Model
   - Overview + Genre + RAG (Boosted)
   - Overview + Genre (Boosted)
3. **Rich Movie Data**: Posters, genres, overviews, and original taglines
4. **Responsive Design**: Works on desktop and mobile


## File Structure

```
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile          # Container configuration
│   ├── deploy.sh           # Cloud Run deployment script
│   └── *.csv              # Data files
├── frontend/
│   ├── app/
│   │   ├── page.tsx        # Main application page
│   │   ├── layout.tsx      # Root layout
│   │   └── globals.css     # Global styles
│   ├── package.json        # Node.js dependencies
│   ├── next.config.js      # Next.js configuration
│   ├── tailwind.config.js  # Tailwind CSS configuration
│   └── vercel.json        # Vercel deployment config
└── setup-dev.sh           # Development setup script
```

## Troubleshooting

1. **CORS Issues**: Make sure your Cloud Run URL is added to CORS origins in `main.py`
2. **Image Loading**: Check that image domains are configured in `next.config.js`
3. **API Connection**: Verify `NEXT_PUBLIC_API_URL` is set correctly in Vercel
4. **CSV Files**: Ensure all CSV files are copied to the backend directory

## Performance Optimizations

- **Backend**: LRU caching for data loading, async endpoints
- **Frontend**: Next.js image optimization, lazy loading
- **Deployment**: Cloud Run auto-scaling, Vercel edge deployment

## Support

For issues or questions:
1. Check the FastAPI docs at `/docs` endpoint
2. Verify all CSV files are present in backend directory
3. Check browser console for frontend errors
4. Ensure environment variables are set correctly
