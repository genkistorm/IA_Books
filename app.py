import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, base64
from scipy.sparse import load_npz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components

# --- 1. CONFIGURATION ET STYLE CSS ---
st.set_page_config(page_title="Stormy AI", page_icon="⚡", layout="centered")

st.markdown("""
<style>
    @keyframes logo-pulse {
        0% { transform: scale(1); opacity: 0.8; }
        50% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(1); opacity: 0.8; }
    }
    .stormy-container { display: flex; justify-content: center; align-items: center; height: 100px; margin-top: -30px; margin-bottom: 20px; }
    .stormy-text { font-family: 'Inter', sans-serif; font-size: 64px; font-weight: 800; background: linear-gradient(90deg, #4F8BF9 0%, #BC67FB 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: logo-pulse 3s infinite ease-in-out; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="stormy-container"><div class="stormy-text">Stormy</div></div>', unsafe_allow_html=True)

# --- 2. GESTION SÉCURISÉE DES AVATARS ---
dossier_actuel = os.path.dirname(os.path.abspath(__file__))
AI_ICON = os.path.join(dossier_actuel, "stormy_icon.png")
USER_ICON = os.path.join(dossier_actuel, "user_icon.png")
AI_AVATAR = AI_ICON if os.path.exists(AI_ICON) else "🤖"
USER_AVATAR = USER_ICON if os.path.exists(USER_ICON) else "👤"

# --- 2b. PHOTO DE PROFIL PERSONNALISÉE (sidebar) ---
# Charger l'avatar par défaut en base64
if os.path.exists(USER_ICON):
    with open(USER_ICON, "rb") as f:
        default_avatar_b64 = base64.b64encode(f.read()).decode()
    default_avatar_src = f"data:image/png;base64,{default_avatar_b64}"
else:
    default_avatar_src = ""

with st.sidebar:
    st.markdown("### Mon profil")

    profile_pic_html = f"""
    <div style="display:flex; justify-content:center; margin-bottom:20px;">
        <div id="profile-wrapper" style="position:relative; width:100px; height:100px; cursor:pointer;" onclick="document.getElementById('profile-upload').click()">
            <img id="profile-img" src="{default_avatar_src}"
                 style="width:100px; height:100px; border-radius:50%; object-fit:cover; border:3px solid #4F8BF9; background:#2a2a2a;"
                 onerror="this.style.display='none'; document.getElementById('profile-placeholder').style.display='flex';" />
            <div id="profile-placeholder" style="display:none; width:100px; height:100px; border-radius:50%;
                 background:linear-gradient(135deg, #4F8BF9, #BC67FB); align-items:center; justify-content:center;
                 font-size:40px; color:white; border:3px solid #4F8BF9;">👤</div>
            <div style="position:absolute; bottom:2px; right:2px; width:28px; height:28px; border-radius:50%;
                 background:linear-gradient(135deg, #4F8BF9, #BC67FB); display:flex; align-items:center;
                 justify-content:center; font-size:16px; font-weight:bold; color:white; border:2px solid #1a1a2e;
                 box-shadow: 0 2px 8px rgba(79,139,249,0.4); transition: transform 0.2s;"
                 onmouseover="this.style.transform='scale(1.15)'" onmouseout="this.style.transform='scale(1)'">+</div>
            <input type="file" id="profile-upload" accept="image/*" style="display:none;" />
        </div>
    </div>
    <script>
        // Charger la photo sauvegardée au démarrage
        (function() {{
            var saved = localStorage.getItem('stormy_profile_pic');
            if (saved) {{
                document.getElementById('profile-img').src = saved;
                document.getElementById('profile-img').style.display = 'block';
                document.getElementById('profile-placeholder').style.display = 'none';
            }}
        }})();

        document.getElementById('profile-upload').addEventListener('change', function(e) {{
            var file = e.target.files[0];
            if (!file) return;
            var reader = new FileReader();
            reader.onload = function(ev) {{
                var dataUrl = ev.target.result;
                // Redimensionner pour ne pas surcharger localStorage
                var img = new Image();
                img.onload = function() {{
                    var canvas = document.createElement('canvas');
                    var size = 200;
                    canvas.width = size;
                    canvas.height = size;
                    var ctx = canvas.getContext('2d');
                    // Crop carré centré
                    var sx=0, sy=0, sw=img.width, sh=img.height;
                    if (sw > sh) {{ sx = (sw-sh)/2; sw = sh; }}
                    else {{ sy = (sh-sw)/2; sh = sw; }}
                    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, size, size);
                    var resized = canvas.toDataURL('image/jpeg', 0.8);
                    localStorage.setItem('stormy_profile_pic', resized);
                    document.getElementById('profile-img').src = resized;
                    document.getElementById('profile-img').style.display = 'block';
                    document.getElementById('profile-placeholder').style.display = 'none';
                }};
                img.src = dataUrl;
            }};
            reader.readAsDataURL(file);
        }});
    </script>
    """
    components.html(profile_pic_html, height=150)

# --- 3. CHARGEMENT DES RESSOURCES (RECOLLAGE TECHNIQUE) ---
@st.cache_resource
def load_resources():
    st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    df = pd.read_csv(os.path.join(dossier_actuel, "data_checkpoint.csv"), encoding='utf-8-sig')
    
    import io

    # Recollage KNN
    knn_data = io.BytesIO()
    for i in range(1, 11):
        p = os.path.join(dossier_actuel, f"moteur_knn_part{i}.pkl")
        if os.path.exists(p):
            with open(p, 'rb') as f: knn_data.write(f.read())
    knn_data.seek(0)
    knn = joblib.load(knn_data)

    # Recollage Memory
    embs_data = io.BytesIO()
    for i in range(1, 5):
        p = os.path.join(dossier_actuel, f"ia_memory_part{i}.npy")
        if os.path.exists(p):
            with open(p, 'rb') as f: embs_data.write(f.read())
    embs_data.seek(0)
    embs = np.load(embs_data)

    # Recollage Matrice
    h_mat_data = io.BytesIO()
    for i in range(1, 5):
        p = os.path.join(dossier_actuel, f"matrice_hybride_part{i}.npz")
        if os.path.exists(p):
            with open(p, 'rb') as f: h_mat_data.write(f.read())
    h_mat_data.seek(0)
    h_mat = load_npz(h_mat_data)

    def fix_encoding(text):
        if not isinstance(text, str): return text
        try: return text.encode('cp1252').decode('utf-8')
        except: return text
    df['Book-Title'] = df['Book-Title'].apply(fix_encoding)

    return df, h_mat, knn, st_model, embs

# --- L'APPEL QUI DÉFINIT 'df' GLOBALEMENT ---
with st.spinner('Chargement de Stormy...'):
    df, h_mat, knn, st_model, embs = load_resources()

# --- 4. INITIALISATION DU CHAT (Tes textes originaux) ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Salut ! Je suis **Stormy**. Je suis capable de te recommander des livres en fonction de tes goûts ! Pour commencer, dis-moi quel est ton livre préféré (ou un livre que tu as aimé récemment) (**de préférence en anglais**)."}]
if "step" not in st.session_state:
    st.session_state.step = "ASK_TITLE"
if "temp_data" not in st.session_state:
    st.session_state.temp_data = {"title": "", "author": "", "count": 8, "diversify": False}

for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- 5. LOGIQUE CONVERSATIONNELLE (Tes textes originaux) ---
if prompt := st.chat_input("Réponds ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AI_AVATAR):
        if st.session_state.step == "ASK_TITLE":
            st.session_state.temp_data["title"] = prompt
            response = f"D'accord, **{prompt}** j'en prends note. Connais-tu son auteur ? (Sinon, réponds 'non' ou laisse un espace vide)"
            st.session_state.step = "ASK_AUTHOR"
            
        elif st.session_state.step == "ASK_AUTHOR":
            st.session_state.temp_data["author"] = "" if prompt.lower() in ["non", "nan", ""] else prompt
            response = "Combien de livres souhaites-tu te voir recommander ? (1 à 10)"
            st.session_state.step = "ASK_COUNT"

        elif st.session_state.step == "ASK_COUNT":
            try:
                count = int(''.join(filter(str.isdigit, prompt)))
                count = max(1, min(10, count))
            except: count = 8
            st.session_state.temp_data["count"] = count
            
            # Dialogue conservé tel quel
            response = "Veux-tu **diversifier** les auteurs ?"
            st.session_state.step = "ASK_DIVERSITY"

        elif st.session_state.step == "ASK_DIVERSITY":
            # MODIFICATION ICI : On accepte o, oui et ouais
            st.session_state.temp_data["diversify"] = prompt.lower() in ['o', 'oui', 'ouais']
            
            title_in = st.session_state.temp_data["title"]
            auth_in = st.session_state.temp_data["author"]
            count = st.session_state.temp_data["count"]
            div = st.session_state.temp_data["diversify"]

            m = df[df['Book-Title'].str.contains(title_in, case=False, na=False)].copy()
            if auth_in:
                m = m[m['Book-Author'].str.contains(auth_in, case=False, na=False)]

            if not m.empty:
                target_row = m.iloc[0]
                idx_pos = target_row.name
                dist, ind = knn.kneighbors(h_mat.getrow(idx_pos), n_neighbors=100)
                
                response = f"Analyse pour : {title_in.upper()}\n\n"
                response += f"Voici {count} ouvrages qui devraient te plaire :\n\n"
                
                target_title = target_row['Book-Title'].lower()
                def clean_auth(name): return "".join(filter(str.isalpha, str(name).lower()))
                t_auth_c = clean_auth(target_row['Book-Author'])
                t_kw = [w for w in target_title.replace("(", "").replace(")", "").split() if len(w) > 3]

                seen_titles = [target_title[:20]]
                mots_interdits = ["audio", "cd", "cassette", "sound recording", "talking book"]
                found = 0
                
                for i in range(1, len(ind[0])):
                    if found >= count: break
                    res = df.iloc[ind[0][i]]
                    titre_propre = res['Book-Title'].lower()
                    res_author = str(res['Book-Author'])
                    
                    if not any(mot in titre_propre for mot in mots_interdits):
                        base_title_short = titre_propre[:20]
                        if base_title_short not in seen_titles:
                            # Filtre Diversité appliqué si div est True
                            if div:
                                if clean_auth(res_author) in t_auth_c or t_auth_c in clean_auth(res_author):
                                    continue
                                if any(k in titre_propre for k in t_kw):
                                    continue

                            found += 1
                            response += f"{found}. **{res['Book-Title']}** ({res['Book-Author']})\n"
                            seen_titles.append(base_title_short)
                
                response += "\n\nDonne-moi un nouveau titre si t'as envie de découvrir d'autres ouvrages !"
                st.session_state.step = "ASK_TITLE"
            else:
                response = f"Je n'ai pas trouvé '{title_in}'. Peux-tu m'en dire un peu plus en me citant plusieurs mots-clés ?"
                st.session_state.step = "ASK_SUMMARY"

        elif st.session_state.step == "ASK_SUMMARY":
            user_title = st.session_state.temp_data["title"]
            count = st.session_state.temp_data["count"]
            nouveau_emb = st_model.encode([f"{user_title} | {prompt}"])
            scores = cosine_similarity(nouveau_emb, embs)[0]
            top_indices = np.argsort(scores)[::-1]
            
            response = f"Analyse pour : {user_title.upper()} ---\n\n"
            seen_titles = [user_title.lower()[:20]]
            mots_interdits = ["audio", "cd", "cassette", "sound recording", "talking book"]
            found = 0
            for idx in top_indices:
                if found >= count: break
                info = df.iloc[idx]
                titre_propre = info['Book-Title'].lower()
                
                if not any(mot in titre_propre for mot in mots_interdits):
                    base_title_short = titre_propre[:20]
                    if base_title_short not in seen_titles:
                        found += 1
                        response += f"{found}. **{info['Book-Title']}** ({info['Book-Author']})\n"
                        seen_titles.append(base_title_short)
            
            response += "\n\nOn continue ? Donne-moi un nouveau titre si t'as envie de découvrir d'autres ouvrages !"
            st.session_state.step = "ASK_TITLE"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
