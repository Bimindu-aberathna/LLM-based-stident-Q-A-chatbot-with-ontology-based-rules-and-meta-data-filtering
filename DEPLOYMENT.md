# Railway Deployment Guide

## Deploy to Railway

1. **Sign up at [Railway.app](https://railway.app/)**

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Connect your repository

3. **Configure Environment Variables**
   - Add `OPENAI_API_KEY` in Railway dashboard
   - Add any other environment variables from your `.env`

4. **Add Build Command (in Railway settings)**
   ```bash
   pip install -r requirements.txt && python -m spacy download en_core_web_sm
   ```

5. **Start Command** (should auto-detect from Procfile)
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

6. **Add Volume for ChromaDB** (Optional, to persist database)
   - Go to "Data" tab
   - Add volume mounted at `/app/data`

## Alternative: Deploy to Render

1. **Sign up at [Render.com](https://render.com/)**

2. **Create New Web Service**
   - Connect your GitHub repository
   - Select Python environment

3. **Configure Build & Start**
   - Build Command: `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Add Environment Variables**
   - Add `OPENAI_API_KEY`
   - Add other variables from `.env`

5. **Note**: Free tier spins down after 15 minutes of inactivity

## Alternative: Deploy to Fly.io

1. **Install Fly CLI**
   ```powershell
   powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
   ```

2. **Login and Initialize**
   ```bash
   fly auth login
   fly launch
   ```

3. **Add Secrets**
   ```bash
   fly secrets set OPENAI_API_KEY=your_key_here
   ```

4. **Deploy**
   ```bash
   fly deploy
   ```

## Memory Optimization Tips

If you hit memory limits on free tiers:

1. **Use OpenAI embeddings only** (remove sentence-transformers if not needed)
2. **Reduce ChromaDB cache**
3. **Lazy load spacy models**
4. **Consider serverless alternatives** (AWS Lambda, Vercel Serverless Functions)

## Cost Monitoring

- Railway: $5/month credit (usually enough for development)
- Render: Free 750 hours/month
- Fly.io: Free 3 VMs, 3GB storage

All costs beyond free tier are pay-as-you-go.
