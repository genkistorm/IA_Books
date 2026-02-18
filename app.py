import streamlit as st
import pandas as pd
import numpy as np
import joblib, os
from scipy.sparse import load_npz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

# --- 2. GESTION DES AVATARS ---
dossier_actuel = os.path.dirname(os.path.abspath(__file__))
AI_ICON = os.path.join(dossier_actuel, "stormy_icon.png")
USER_ICON = os.path.join(dossier_actuel, "user_icon.png")
AI_AVATAR = AI_ICON if os.path.exists(AI_ICON) else "🤖"
USER_AVATAR = USER_ICON if os.path.exists(USER_ICON) else "👤"

# --- 3. CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_resources():
    st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    df = pd.read_csv(os.path.join(dossier_actuel, "data_checkpoint.csv"), encoding='utf-8-sig')
    df = df.reset_index(drop=True)
    
    import io
    def glue_to_disk(prefix, count, extension):
        temp_path = os.path.join(dossier_actuel, f"temp_{prefix}.{extension}")
        with open(temp_path, 'wb') as f_out:
            for i in range(1, count + 1):
                p = os.path.join(dossier_actuel, f"{prefix}_part{i}.{extension}")
                if os.path.exists(p):
                    with open(p, 'rb') as f_in: f_out.write(f_in.read())
        return temp_path

    knn = joblib.load(glue_to_disk("moteur_knn", 10, "pkl"))
    embs = np.load(glue_to_disk("ia_memory", 5, "npy"))
    h_mat = load_npz(glue_to_disk("matrice_hybride", 5, "npz"))

    def fix_encoding(text):
        if not isinstance(text, str): return str(text)
        try: return text.encode('cp1252').decode('utf-8')
        except: return text
    df['Book-Title'] = df['Book-Title'].apply(fix_encoding)
    df['Book-Author'] = df['Book-Author'].apply(fix_encoding)

    return df, h_mat, knn, st_model, embs

with st.spinner('Chargement de Stormy...'):
    df, h_mat, knn, st_model, embs = load_resources()

# --- 4. INITIALISATION DU CHAT ---
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

# --- 5. LOGIQUE CONVERSATIONNELLE ---
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
            response = "Veux-tu **diversifier** les auteurs ?"
            st.session_state.step = "ASK_DIVERSITY"

        elif st.session_state.step == "ASK_DIVERSITY":
            st.session_state.temp_data["diversify"] = prompt.lower() in ['o', 'oui', 'ouais']
            title_in = st.session_state.temp_data["title"]
            auth_in = st.session_state.temp_data["author"]
            count = st.session_state.temp_data["count"]
            div = st.session_state.temp_data["diversify"]

            # Priorisation des titres plus courts pour éviter de matcher les "Companions" en premier
            m = df[df['Book-Title'].str.contains(title_in, case=False, na=False)].copy()
            if auth_in:
                m = m[m['Book-Author'].str.contains(auth_in, case=False, na=False)]
            
            if not m.empty:
                m['len'] = m['Book-Title'].str.len()
                target_row = m.sort_values('len').iloc[0]
                idx_pos = target_row.name 
                
                # Scan ultra-profond (2000 voisins) pour ne rien rater
                dist, ind = knn.kneighbors(h_mat.getrow(idx_pos), n_neighbors=min(2000, len(df)))
                
                response = f"Analyse pour : {target_row['Book-Title'].upper()}\n\n"
                response += f"Voici {count} ouvrages qui devraient te plaire :\n\n"
                
                t_title = str(target_row['Book-Title']).lower()
                t_auth = str(target_row['Book-Author'])
                def clean_auth(name): return "".join(filter(str.isalpha, str(name).lower()))
                t_auth_clean = clean_auth(t_auth)
                t_kw = [w for w in t_title.replace("(", "").replace(")", "").split() if len(w) > 3]

                seen_titles = [t_title[:20]]
                found = 0
                
                for i in range(1, len(ind[0])):
                    if found >= count: break
                    res = df.iloc[ind[0][i]]
                    res_title = str(res['Book-Title']).lower()
                    res_auth_clean = clean_auth(str(res['Book-Author']))
                    
                    if res_title[:20] in seen_titles: continue

                    if div:
                        # Si OUI : on exclut l'auteur original
                        if res_auth_clean in t_auth_clean or t_auth_clean in res_auth_clean: continue
                        if any(k in res_title for k in t_kw): continue
                    else:
                        # Si NON : on n'accepte QUE le même auteur au début
                        if res_auth_clean not in t_auth_clean and t_auth_clean not in res_auth_clean:
                            # Si on a déjà scanné beaucoup de livres et qu'on n'a rien, on laisse le KNN
                            # proposer les plus proches sémantiquement pour ne pas finir vide
                            if i < 500: continue 

                    found += 1
                    response += f"{found}. **{res['Book-Title']}** ({res['Book-Author']})\n"
                    seen_titles.append(res_title[:20])
                
                response += "\n\nDonne-moi un nouveau titre si t'as envie de découvrir d'autres ouvrages !"
                st.session_state.step = "ASK_TITLE"
            else:
                response = f"Je n'ai pas trouvé '{title_in}'. Peux-tu m'en dire un peu plus en me citant plusieurs mots-clés ?"
                st.session_state.step = "ASK_SUMMARY"

        elif st.session_state.step == "ASK_SUMMARY":
            user_title = st.session_state.temp_data["title"]
            count = st.session_state.temp_data["count"]
            nouveau_feat = f"Fantasy Fantasy | {user_title} | {prompt}"
            nouveau_emb = st_model.encode([nouveau_feat])
            scores = cosine_similarity(nouveau_emb, embs)[0]
            top_indices = np.argsort(scores)[::-1]
            
            response = f"Analyse pour : {user_title.upper()} ---\n\n"
            seen = [user_title.lower()[:20]]
            found = 0
            for idx in top_indices:
                if found >= count: break
                info = df.iloc[idx]
                if str(info['Book-Title']).lower()[:20] not in seen:
                    found += 1
                    response += f"{found}. **{info['Book-Title']}** ({info['Book-Author']})\n"
                    seen.append(str(info['Book-Title']).lower()[:20])
            
            response += "\n\nOn continue ? Donne-moi un nouveau titre si t'as envie de découvrir d'autres ouvrages !"
            st.session_state.step = "ASK_TITLE"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})