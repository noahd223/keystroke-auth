# 🚀 Render Deployment Guide

This guide will walk you through deploying your Keystroke Authenticator full stack app on Render.

## 📋 Prerequisites

1. **GitHub Account**: Your code needs to be in a GitHub repository
2. **Render Account**: Sign up at [render.com](https://render.com)
3. **Git**: Make sure your project is committed and pushed to GitHub

## 🔧 Configuration Files Created

The following files have been created/modified for Render deployment:

### Backend Configuration:
- `back-end/requirements.txt` - Updated with CPU-only PyTorch and gunicorn
- `back-end/collection.py` - Updated with environment variables and health check
- `back-end/render.yaml` - Render service configuration (optional)

### Frontend Configuration:
- `front-end/.env.production` - Production API URL for Render
- `front-end/.env.local` - Local development API URL
- `front-end/package.json` - Added serve package for static hosting

## 🚀 Step-by-Step Deployment

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Add Render deployment configuration"
git push origin main
```

### Step 2: Deploy Backend (Flask API)

1. **Go to Render Dashboard**
   - Visit [render.com](https://render.com)
   - Click "Get Started for Free" and sign up with GitHub
   - Click "New +" → "Web Service"

2. **Connect Repository**
   - Select "Build and deploy from a Git repository"
   - Click "Connect" next to your `keystroke-authenticator` repository
   - If not visible, click "Configure account" to grant access

3. **Configure Backend Service**
   - **Name**: `keystroke-auth-backend`
   - **Root Directory**: `back-end`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT collection:app`

4. **Set Environment Variables**
   - Scroll down to "Environment Variables"
   - Add these variables:
     ```
     FLASK_ENV = production
     PYTHONPATH = /opt/render/project/src
     ```

5. **Choose Plan**
   - Select "Free" plan (0.1 CPU, 512MB RAM)
   - Click "Create Web Service"

6. **Wait for Deployment**
   - Build will take 5-10 minutes due to PyTorch
   - Note your backend URL: `https://keystroke-auth-backend.onrender.com`

### Step 3: Deploy Frontend (React App)

1. **Create Another Web Service**
   - Go back to Render dashboard
   - Click "New +" → "Web Service"
   - Connect the same repository

2. **Configure Frontend Service**
   - **Name**: `keystroke-auth-frontend`
   - **Root Directory**: `front-end`
   - **Environment**: `Node`
   - **Build Command**: `npm install && npm run build`
   - **Start Command**: `npx serve -s build -l $PORT`

3. **Set Environment Variables**
   - Add this variable:
     ```
     REACT_APP_API_URL = https://keystroke-auth-backend.onrender.com
     ```
   - Replace with your actual backend URL from Step 2

4. **Choose Plan & Deploy**
   - Select "Free" plan
   - Click "Create Web Service"
   - Wait for deployment to complete

## 🔧 Alternative: Manual Deployment Steps

If you prefer manual setup instead of using render.yaml:

### Backend Manual Setup:
1. New Web Service → Connect repo
2. Root Directory: `back-end`
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn --bind 0.0.0.0:$PORT collection:app`
5. Environment Variables:
   - `FLASK_ENV=production`
   - `PYTHONPATH=/opt/render/project/src`

### Frontend Manual Setup:
1. New Web Service → Connect repo
2. Root Directory: `front-end`
3. Build Command: `npm install && npm run build`
4. Start Command: `npx serve -s build -l $PORT`
5. Environment Variables:
   - `REACT_APP_API_URL=https://your-backend-url.onrender.com`

## 🌐 Custom Domain (Optional)

1. Go to your service settings
2. Click "Custom Domains"
3. Add your domain
4. Update DNS records as instructed
5. Render provides free SSL certificates

## 🔍 Troubleshooting

### Common Issues:

1. **Build Timeout**: 
   - Render free tier has 15-minute build limit
   - PyTorch is large but should fit within limit
   - Consider upgrading to paid plan if needed

2. **Memory Issues**: 
   - Free tier: 512MB RAM
   - Should be sufficient for basic ML models
   - Monitor usage in Render dashboard

3. **Cold Starts**: 
   - Free tier services sleep after 15 minutes of inactivity
   - First request after sleep takes ~30 seconds
   - Paid plans don't have this limitation

4. **CORS Issues**:
   - Ensure backend URL is correct in frontend env vars
   - Check CORS configuration in Flask app

5. **File Persistence**:
   - Render's filesystem is ephemeral
   - Use Render PostgreSQL or external storage for persistent data

### Logs and Debugging:
- View logs in Render service dashboard
- Use health check: `https://your-backend-url.onrender.com/health`
- Check "Events" tab for deployment issues

## 💰 Cost Considerations

**Render Free Tier:**
- 750 hours/month per service
- Services sleep after 15 minutes of inactivity
- 512MB RAM, 0.1 CPU
- Perfect for development/testing

**Paid Plans:**
- Starter: $7/month per service
- No sleeping, better performance
- More RAM and CPU options

## 🎉 Success!

Once deployed, you'll have:
- **Backend API**: `https://keystroke-auth-backend.onrender.com`
- **Frontend App**: `https://keystroke-auth-frontend.onrender.com`
- **Health Check**: `https://keystroke-auth-backend.onrender.com/health`

Your keystroke authenticator is now live! 🌍

## 📝 Key Differences from Railway

**Advantages of Render:**
- ✅ More generous free tier (750 hours vs 500 hours)
- ✅ Simpler pricing structure
- ✅ Better documentation and UI
- ✅ Free SSL certificates
- ✅ Built-in PostgreSQL option

**Considerations:**
- ⚠️ Services sleep on free tier (15 min inactivity)
- ⚠️ Cold start delay (~30 seconds)
- ⚠️ Less flexibility than Railway for complex deployments

## 🔄 Migration from Railway

If you had Railway setup before:
1. All Railway config files have been removed
2. New Render-specific files created
3. Environment variables updated for Render URLs
4. Ready to deploy on Render!

## 📞 Support

- **Render Docs**: [render.com/docs](https://render.com/docs)
- **Community**: Render Discord server
- **Status**: [status.render.com](https://status.render.com)

---

**Ready to deploy?** Follow the steps above and your app will be live in minutes! 🚀 