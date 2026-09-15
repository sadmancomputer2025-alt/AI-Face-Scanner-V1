# AI Face Scanner

> **Real-Time Browser-Based AI Face Detection & Analysis**  
> **Developed by Sadman Sakib**  
> *© 2026 Sadman Sakib — AI Face Scanner*

[![Live Demo](https://img.shields.io/badge/Demo-Live_Preview-0284c7?style=for-the-badge&logo=googlechrome&logoColor=white)](https://github.com/)
[![Built with JavaScript](https://img.shields.io/badge/Language-Vanilla_JavaScript_(ES6+)-f7df1e?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![AI Engine](https://img.shields.io/badge/AI_Engine-face--api.js_&_TensorFlow.js-ff6f00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://github.com/justadudewhohacks/face-api.js)
[![Privacy](https://img.shields.io/badge/Privacy-100%25_On--Device-10b981?style=for-the-badge&logo=auth0&logoColor=white)](#privacy-first-architecture)

---

## 🌟 Overview

**AI Face Scanner** is a modern, high-performance, browser-based real-time facial detection and analysis web application. Designed with a SaaS-grade dashboard aesthetic, it processes webcam video feeds and photos entirely within the user's browser using deep convolutional neural networks.

Inspired by classic prototype face detectors, this project elevates the concept into a production-ready, accessible, and responsive tool featuring a clean white and light-blue visual theme, glassmorphic elements, cybernetic HUD overlays, and real-time biometric analytics.

---

## ✨ Key Features

- 📸 **Live Webcam Scanning:** Instant WebRTC camera connection with 16:9 responsive aspect ratio container.
- 👥 **Multi-Person Real-Time Detection:** Tracks single or multiple faces concurrently with zero lag.
- 🎯 **Accurate Cyan/Blue Bounding Boxes:** Dynamic bounding boxes with high-tech corner brackets and floating header tags (`Person 1 • 24 yrs • Male (98%) • 😄 Happy`).
- 🎂 **Age Estimation:** Continuous neural age regression with life-stage classification (*Child, Teen, Young Adult, Adult, Senior*).
- ⚧️ **Gender Classification:** Gender prediction with live probability meter.
- 🙂 **Affective Computing & Micro-Expressions:** 7-emotion softmax classification (*Neutral, Happy, Sad, Angry, Fearful, Disgusted, Surprised*) with micro-expression breakdown bars.
- 📐 **68-Point Facial Landmark Mesh:** Toggleable luminous neural mesh visualizing facial contours, jawline, nose ridge, brows, and lips.
- 📊 **Dynamic Statistics Dashboard:** Automatically generates and animates individual person cards as individuals enter or exit the camera view.
- 🧪 **Offline Sample Photo Tester:** Includes built-in single portrait, duo selfie, and group test photos, plus custom image file upload—enabling AI testing on devices without webcams.
- 📸 **AI Snapshot Tool:** Capture and download high-resolution annotated frames with AI tags and watermarks.
- 🛡️ **Zero Cloud Inference:** 100% on-device WebGL accelerated execution. Zero video bytes leave your machine.

---

## 🧠 AI Neural Models

AI Face Scanner utilizes four specialized deep learning models powered by `face-api.js` and `TensorFlow.js`:

1. **Tiny Face Detector:** MobileNetV1-based real-time face detector utilizing depthwise separable convolutions for high inference speed and low power consumption.
2. **Face Landmark 68 Net:** Predicts 68 2D spatial landmark points mapping eyebrows, eyes, nose, mouth, and jawline contours.
3. **Age & Gender Recognition Net:** Multi-task CNN that regresses biological age (in years) and predicts binary gender with confidence scoring.
4. **Face Expression Recognition Net:** Computes softmax probability distribution across 7 universal facial emotion states.

---

## 🔒 Privacy First Architecture

- **No Remote Streaming:** Video frames from `navigator.mediaDevices.getUserMedia` are passed strictly to WebGL textures in browser memory.
- **No Storage:** No biometric signatures, snapshots, or recordings are saved to `localStorage`, cookies, or remote databases.
- **Immediate Resource Release:** Pressing "Stop Camera" terminates all `MediaStream` hardware tracks and immediately frees the webcam.

> **Disclaimer:** Age, gender, and expression results are AI-generated statistical estimates and should not be used for identification or verification purposes.

---

## 💻 Tech Stack & Architecture

- **Frontend:** HTML5, Modern CSS3 (CSS Variables, Flexbox, CSS Grid), Vanilla ES6+ JavaScript.
- **AI / Computer Vision:** `face-api.js` (v0.22.2) on top of `tfjs-core` (WebGL backend).
- **Deployment:** Zero dependencies, static site ready for GitHub Pages, Netlify, Vercel, or Apache/Nginx.

---

## 🚀 Quick Start & Deployment

### 1. Run Locally
Clone the repository and serve the static files using any local web server:

```bash
# Clone the repository
git clone https://github.com/your-username/ai-face-scanner.git
cd ai-face-scanner

# Serve with Python 3
python3 -m http.server 3000

# OR serve with Node.js npx
npx serve .
```

Open `http://localhost:3000` in your browser.

### 2. Deploy to GitHub Pages
1. Push this repository to GitHub.
2. Go to **Settings** > **Pages**.
3. Under **Build and deployment**, select **Source: Deploy from a branch**.
4. Choose the `main` branch and `/ (root)` folder, then click **Save**.
5. Your live app will be published at `https://<username>.github.io/<repo-name>/`.

---

## 👨‍💻 Developer Attribution

- **Project:** AI Face Scanner
- **Developed by:** **Sadman Sakib**
- **Copyright:** © 2026 Sadman Sakib — AI Face Scanner

All rights reserved.
