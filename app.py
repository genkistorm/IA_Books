import streamlit as st
import pandas as pd
import numpy as np
import joblib, os
from scipy.sparse import load_npz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Stormy AI", page_icon="⚡", layout="centered")

st.markdown("""
<style>
    /* Effet de pulsation du logo */
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

# --- 2. CHEMIN DU RÉPERTOIRE ---
dossier_actuel = os.path.dirname(os.path.abspath(__file__))

# --- 3. CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_resources():
    prompt_titre = ""
    st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    df = pd.read_csv(os.path.join(dossier_actuel, "data_checkpoint.csv"), encoding='utf-8-sig')
    df = df.reset_index(drop=True)
    
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
        try:
            return text.encode('cp1252').decode('utf-8').encode('cp1252').decode('utf-8')
        except:
            try:
                return text.encode('cp1252').decode('utf-8')
            except:
                return text
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
    avatar = "stormy_icon.png" if message["role"] == "assistant" else "user_icon.png"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- 5. LOGIQUE CONVERSATIONNELLE ---
if prompt := st.chat_input("Réponds ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="user_icon.png"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="stormy_icon.png"):
        if st.session_state.step == "ASK_TITLE":
            st.session_state.temp_data["title"] = prompt
            prompt_titre = prompt
            response = f"D'accord, **{prompt}** j'en prends note. Connais-tu son auteur ? (Sinon, réponds 'non' ou laisse un espace vide)"
            st.session_state.step = "ASK_AUTHOR"
            
        elif st.session_state.step == "ASK_AUTHOR":
            st.session_state.temp_data["author"] = "" if prompt.lower() in ["non", "nan", ""] else prompt
            response = "Combien de livres souhaites-tu te voir recommander ? (1 à 10)"
            st.session_state.step = "ASK_COUNT"

        elif st.session_state.step == "ASK_COUNT":
            digits = ''.join(filter(str.isdigit, prompt))
            if digits:
                count = int(digits)
                if 1 <= count <= 10:
                    st.session_state.temp_data["count"] = count
                    response = "Veux-tu **diversifier** les propositions ?"
                    st.session_state.step = "ASK_DIVERSITY"
                else:
                    response = "Désolé, je ne peux recommander qu'entre **1 et 10 livres**. Peux-tu me donner un chiffre valide ?"
            else:
                response = "Oups ! Je n'ai pas vu de chiffre. Peux-tu me donner un chiffre entre **1 et 10** ?"

        elif st.session_state.step == "ASK_DIVERSITY":
            st.session_state.temp_data["diversify"] = prompt.lower() in ['o', 'oui', 'ouais']
            title_in = st.session_state.temp_data["title"]
            auth_in = st.session_state.temp_data["author"]
            count = st.session_state.temp_data["count"]
            div = st.session_state.temp_data["diversify"]

            m = df[df['Book-Title'].str.contains(title_in, case=False, na=False)].copy()
            if not m.empty:
                if auth_in and len(auth_in) > 1:
                    m_auth = m[m['Book-Author'].str.contains(auth_in, case=False, na=False)]
                    if not m_auth.empty: m = m_auth

                m['len'] = m['Book-Title'].str.len()
                target_row = m.sort_values('len').iloc[0]
                idx_pos = target_row.name 
                dist, ind = knn.kneighbors(h_mat.getrow(idx_pos), n_neighbors=min(1000, len(df)))
                
                response = f"Analyse pour : {prompt_titre}\n\n"
                response += f"Voici {count} ouvrages qui devraient te plaire :\n\n"
                
                t_title = str(target_row['Book-Title']).lower()
                t_auth = str(target_row['Book-Author'])
                def clean_auth(name): return "".join(filter(str.isalpha, str(name).lower()))
                t_auth_c = clean_auth(t_auth)
                t_kw = [w for w in t_title.replace("(", "").replace(")", "").split() if len(w) > 3]

                seen_titles = [t_title[:20]]
                found = 0
                for i in range(1, len(ind[0])):
                    if found >= count: break
                    res = df.iloc[ind[0][i]]
                    res_title = str(res['Book-Title']).lower()
                    res_auth_c = clean_auth(str(res['Book-Author']))
                    if res_title[:20] in seen_titles: continue
                    if div:
                        if res_auth_c in t_auth_c or t_auth_c in res_auth_c: continue
                        if any(k in res_title for k in t_kw): continue
                    else:
                        if not (res_auth_c in t_auth_c or t_auth_c in res_auth_c or any(k in res_title for k in t_kw)): continue
                    found += 1
                    response += f"{found}. **{res['Book-Title']}** ({res['Book-Author']})\n"
                    seen_titles.append(res_title[:20])
                
                if found == 0: response += "(Pas assez de résultats trouvés. Essaye de diversifier !)"
                response += "\n\nDonne-moi un nouveau titre !"
                st.session_state.step = "ASK_TITLE"
            else:
                response = f"Je n'ai pas trouvé '{title_in}'. Peux-tu m'en dire plus avec des mots-clés ?"
                st.session_state.step = "ASK_SUMMARY"

        elif st.session_state.step == "ASK_SUMMARY":
            user_title = st.session_state.temp_data["title"]
            count = st.session_state.temp_data["count"]
            nouveau_feat = f"{user_title} | {prompt}"
            nouveau_emb = st_model.encode([nouveau_feat])
            scores = cosine_similarity(nouveau_emb, embs)[0]
            top_indices = np.argsort(scores)[::-1]
            
            response = f"Analyse pour : {user_title.upper()}\n\n"
            seen, found = [user_title.lower()[:20]], 0
            for idx in top_indices:
                if found >= count: break
                info = df.iloc[idx]
                if str(info['Book-Title']).lower()[:20] not in seen:
                    found += 1
                    response += f"{found}. **{info['Book-Title']}** ({info['Book-Author']})\n"
                    seen.append(str(info['Book-Title']).lower()[:20])
            response += "\n\nDonne-moi un nouveau titre !"
            st.session_state.step = "ASK_TITLE"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
