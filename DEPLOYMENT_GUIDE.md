# 🚀 Deployment Guide for CIRA on IIoT

## Quick Deploy to Render.com (FREE)

### Step 1: Prepare Your Repository

1. **Push deployment files to GitHub:**
   ```bash
   git add requirements.txt render.yaml Procfile runtime.txt .gitignore
   git commit -m "Add deployment configuration"
   git push origin main
   ```

### Step 2: Deploy on Render

1. **Go to Render.com:**
   - Visit: https://render.com
   - Click "Get Started for Free"
   - Sign up with GitHub

2. **Create New Web Service:**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `rajatmp/ciira`
   - Click "Connect"

3. **Configure Service:**
   - **Name:** `cira-iiot` (or your choice)
   - **Region:** Oregon (Free)
   - **Branch:** `main`
   - **Root Directory:** Leave empty
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn --chdir CODE/APP app:app`
   - **Plan:** Free

4. **Environment Variables (Optional):**
   - Add if needed:
     - `SECRET_KEY`: Generate a random string
     - `FLASK_ENV`: production

5. **Click "Create Web Service"**

### Step 3: Wait for Deployment

- First deployment takes 5-10 minutes
- Watch the logs for any errors
- Once deployed, you'll get a URL like: `https://cira-iiot.onrender.com`

---

## ⚠️ Important Notes

### Free Tier Limitations:
- **Sleeps after 15 minutes** of inactivity
- **Wakes up in ~30 seconds** on first request
- **750 hours/month** (enough for continuous use)
- **512MB RAM** (sufficient for your models)

### Database:
- SQLite works on Render but data resets on restart
- For persistent data, upgrade to PostgreSQL (still free)

### File Uploads:
- Uploaded files are temporary (lost on restart)
- Consider using cloud storage for production

---

## 🔧 Troubleshooting

### Build Fails:
```bash
# Check requirements.txt versions
pip freeze > requirements.txt
```

### App Won't Start:
- Check logs in Render dashboard
- Verify `app:app` points to your Flask app
- Ensure `SECRET_KEY` is set

### Models Not Loading:
- Verify models are in repository
- Check file paths are relative
- Models folder should be committed

---

## 🎯 Alternative: PythonAnywhere

### Steps:
1. Sign up at https://www.pythonanywhere.com
2. Upload your code via Git or Files
3. Create a new web app (Flask)
4. Configure WSGI file to point to `CODE/APP/app.py`
5. Set working directory to `CODE/APP`

### Advantages:
- Always-on (no sleep)
- Simpler setup
- Good for beginners

### Disadvantages:
- Slower performance
- Limited storage (512MB)
- Manual updates

---

## 📝 Post-Deployment Checklist

- [ ] Test login/registration
- [ ] Upload sample CSV
- [ ] Train a model
- [ ] Run predictions
- [ ] Check all pages load
- [ ] Verify dark/light mode works
- [ ] Test file downloads

---

## 🔐 Security for Production

Before going live:

1. **Change SECRET_KEY:**
   ```python
   # In app.py
   app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fallback-key')
   ```

2. **Use Environment Variables:**
   - Don't commit sensitive data
   - Use Render's environment variables

3. **Enable HTTPS:**
   - Render provides free SSL
   - Force HTTPS in production

4. **Database:**
   - Switch to PostgreSQL for persistence
   - Backup regularly

---

## 🌐 Your Deployed App

Once deployed, your app will be available at:
- **Render:** `https://cira-iiot.onrender.com`
- **Custom Domain:** Can be added (requires DNS setup)

Share this URL with users!

---

## 📞 Need Help?

- Render Docs: https://render.com/docs
- Check deployment logs in Render dashboard
- GitHub Issues: https://github.com/rajatmp/ciira/issues
