# Deployment Checklist

## Pre-deployment
- [ ] All CSV files copied to `backend/` directory
- [ ] Google Cloud SDK installed and configured
- [ ] Vercel CLI installed or GitHub repository connected
- [ ] Docker installed (for Cloud Run deployment)

## Backend Deployment (Cloud Run)
- [ ] Update `PROJECT_ID` in `backend/deploy.sh`
- [ ] Run `cd backend && ./deploy.sh`
- [ ] Note the deployment URL from output
- [ ] Test API endpoints: `/`, `/movies`, `/movie/Toy%20Story`

## Frontend Deployment (Vercel)
- [ ] Update `NEXT_PUBLIC_API_URL` with Cloud Run URL
- [ ] Deploy via `cd frontend && vercel` or GitHub integration
- [ ] Test frontend functionality
- [ ] Verify CORS is working (check browser console)

## Post-deployment Testing
- [ ] Select a movie from dropdown
- [ ] Verify movie poster loads
- [ ] Check all generated taglines display
- [ ] Test on mobile device
- [ ] Verify API response times

## Production Configuration
- [ ] Update CORS origins in `backend/main.py` to include Vercel domain
- [ ] Set up custom domain (optional)
- [ ] Configure monitoring/logging
- [ ] Set up SSL certificates (handled by Vercel/Cloud Run automatically)

## Troubleshooting
- [ ] Check Cloud Run logs: `gcloud logs read --service=tagline-generator-api`
- [ ] Check Vercel deployment logs in dashboard
- [ ] Verify environment variables are set correctly
- [ ] Test API endpoints directly in browser or with curl
