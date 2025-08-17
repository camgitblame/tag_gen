#!/bin/bash

# Cloud Run deployment script for Tagline Generator API

# Set your project variables
PROJECT_ID="your-project-id"  # Replace with your GCP project ID
SERVICE_NAME="tagline-generator-api"
REGION="us-central1"  # Change if needed

# Build and deploy to Cloud Run
echo "Building and deploying to Cloud Run..."

gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --max-instances 10

echo "Deployment complete! Your API is available at:"
gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)'
