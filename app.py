import streamlit as st
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import shap

st.set_page_config(page_title="Forensic Image Authenticator", layout="wide")

# تحميل الموديل 
@st.cache_resource
def load_assets():
    try:
        data = joblib.load("model.pkl")
        return data
    except:
        st.error("🚨 ملف model.pkl غير موجود! تأكدي من تشغيل كود التدريب أولاً.")
        st.stop()

assets = load_assets()
final_model = assets["model"]
categories = assets["categories"]
winner_name = assets["model_name"]
X_train_ref = assets.get("X_train", np.zeros((100, 56)))
model_scores = assets.get("scores", {})

def extract_features_app(image):
    # تحويل لرمادي وتصغير
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, (128, 128)) 

    # ميزات LBP (P=24, R=3) -> 26 bins
    lbp = local_binary_pattern(img_resized, P=24, R=3, method='uniform')
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-7)

    # ميزات FFT -> 30 bins
    f_transform = np.fft.fft2(img_resized)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    (fft_hist, _) = np.histogram(magnitude_spectrum.ravel(), bins=30, range=(0, 255))
    fft_hist = fft_hist.astype("float") / (fft_hist.sum() + 1e-7)

    combined = np.hstack([lbp_hist, fft_hist]).reshape(1, -1)
    return combined, lbp, magnitude_spectrum

# ==========================================
# ==========================================
st.sidebar.title("📊 مقارنة النماذج")
st.sidebar.markdown(f"**الموديل الفائز:** {winner_name}")

if model_scores:
    sorted_scores = dict(sorted(model_scores.items(), key=lambda x: x[1], reverse=True))

    score_table = {
        "الموديل": list(sorted_scores.keys()),
      "الدقة (Balanced CV)": [f"{v:.2f}%" for v in sorted_scores.values()]
    }

    st.sidebar.table(score_table)

# ==========================================

# ==========================================
st.title("🛡️ نظام كشف تزوير الصور الرقمية")
st.write("ارفع صورة للتحقق من أصالتها وكشف إعادة التصوير (Recapture Detection).")

uploaded_file = st.file_uploader("اختر صورة...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, channels="BGR", caption="الصورة المرفوعة", use_container_width=True)
        analyze_btn = st.button("بدء التحليل  ")

    if analyze_btn:
        with st.spinner("جاري استخراج الميزات وحساب تفسير SHAP..."):
            # استخراج الميزات والتوقع 
            feats, lbp_viz, fft_viz = extract_features_app(image)
            prediction = final_model.predict(feats)[0]
            probs = final_model.predict_proba(feats)[0]
            
            label = categories[prediction]
            confidence = probs[prediction] * 100

            # حساب SHAP
            explainer = shap.KernelExplainer(final_model.predict_proba, shap.sample(X_train_ref, 20))
            shap_values = explainer.shap_values(feats, nsamples=100)

        with col2:
            if label == "Original":
                st.success("### Result: Original Image ✅")
            else:
                st.error("### Result: Recaptured Image ⚠️")
            

            st.metric("Confidence Score", f"{confidence:.2f}%")

            st.progress(int(confidence))
                    
            st.caption(f"{label} confidence level")
            st.write("---")
            st.write("**Prediction Explanation (XAI):**")

            # Process SHAP 
            if isinstance(shap_values, list):
                sv = np.array(shap_values[prediction]).ravel()
            else:
                sv = np.array(shap_values).ravel()

            #   Feature Naming 
            n_features = len(sv)
            
            half = n_features // 2
            feature_names = [f'LBP_{i}' for i in range(half)] + \
                            [f'FFT_{i}' for i in range(n_features - half)]

            #  أهم 10 ميزات
            k = min(10, n_features)
            indices = np.argsort(np.abs(sv))[-k:]
            
            
            fig_s, ax_s = plt.subplots(figsize=(10, 7))
            current_values = sv[indices]
            current_labels = [feature_names[i] for i in indices]
            
            bar_colors = ['#ff4b4b' if val < 0 else '#00cc96' for val in current_values]
            
            ax_s.barh(range(k), current_values, color=bar_colors)
            ax_s.set_yticks(range(k))
            ax_s.set_yticklabels(current_labels, fontsize=10) 
            
            ax_s.set_title("Top Features Influencing the Prediction", fontsize=14)
            ax_s.set_xlabel("SHAP Value (Impact Intensity)", fontsize=11)
            
            plt.tight_layout()
            st.pyplot(fig_s)
        # =======================================================
        # =======================================================
        st.divider()
        st.header(" التقرير الفني للأدلة المكتشفة")
        t1, t2 = st.tabs(["تحليل النسيج (LBP)", "التحليل الترددي (FFT)"])
        
        with t1:
            f1, a1 = plt.subplots()
            a1.imshow(lbp_viz, cmap='gray')
            a1.set_title("Local Binary Pattern Visualization")
            a1.axis('off')
            st.pyplot(f1)
            st.write("الـ LBP يكشف التغيرات الميكروية في النسيج.")

        with t2:
            f2, a2 = plt.subplots()
            a2.imshow(fft_viz, cmap='magma')
            a2.axis('off')
            st.pyplot(f2)
            st.write("الـ FFT يكشف التداخلات الترددية (Moire Patterns).")