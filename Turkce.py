import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
# aa.py



# Streamlit stil düzenlemeleri
st.markdown(
    """
    <style>
    .stApp {
        background-color: #D3D3D3;    # rgb(0, 0, 255),#rgba(0, 128, 255, 0.5)
        color: #000000;
    }
    .question-textarea {
        background-color: #5cb85c; /* Yeşil arka plan rengi */
        color: black; /* Yazı rengi */
        padding: 8px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .answer-textarea {
        background-color: #5bc0de; /* Mavi arka plan rengi */
        color: black; /* Yazı rengi */
        padding: 8px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .question-label {
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sayfa başlığı ve resim
st.markdown('''
    <div style="text-align: center">
        <p style="background-color: #FFBF00; color: black; font-size: 30px; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.1);">
            🤖 OneAmz Assistant 🤖<br>
            <span style="font-size: 14px; color:green;">Cevrimici</span>
        </p>
    </div>
''', unsafe_allow_html=True)

st.sidebar.image("chat.png")
st.sidebar.markdown("""
    <div style="text-align: left">
        <h3>ChatBot Nedir?</h3>
        <p>Bir ChatBot, metin veya ses aracılığıyla insanlarla iletişim kuran yapay zeka destekli bir programdır.</p>
        <p>Doğal dil işleme ve yapay zeka kullanarak soruları yanıtlar, bilgi sağlar ve görevleri yerine getirir.</p>
        <p>ChatBot'lar geniş bir yelpazede kullanılır, müşteri hizmetleri, eğitim, sağlık hizmetleri ve daha birçok alanda kullanılır.

</p>
    </div>
""", unsafe_allow_html=True) 


# Önceki soru-cevap kayıtları için session_state kontrolü
if 'qa_log' not in st.session_state:
    st.session_state['qa_log'] = []
    
    

# Tokenizer ve modeli yükle
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
model = BertForQuestionAnswering.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")

# Veri setini yükle (xlsx formatında)
@st.cache_data  # Bu fonksiyonun sonucunu önbelleğe al
def load_data(file_path):
    return pd.read_excel(file_path)

data_path = 'kısayollar-2.xlsx'  # Veri setinizin yolu
database = load_data(data_path)

# En uygun bağlamı bulma fonksiyonu
def find_best_context(question, database):
    best_match = ""
    max_matches = 0
    for index, row in database.iterrows():
        paragraph = row['CEVAPLAR']
        matches = sum(1 for word in question.split() if word in paragraph)
        if matches > max_matches:
            max_matches = matches
            best_match = paragraph
    return best_match

# Kullanıcıdan soru alma
question = st.text_input("Soru Sor:")
if not question:
    st.markdown('<div class="question-textarea">Merhabalar, size nasil yardimci olabilirim ?</div>', unsafe_allow_html=True)

# Cevabı hesapla ve göster
if st.button("Cevabi Bul") and question:
    # Soruya en uygun bağlamı bul
    context = find_best_context(question, database)
    
    # Girişi tokenize et
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt', truncation=True, max_length=512)
    
    # Bağlamı kırp
    max_context_length = 300
    if len(inputs['input_ids'][0]) > max_context_length:
        inputs['input_ids'][0] = inputs['input_ids'][0][:max_context_length]
    
    with torch.no_grad():
        outputs = model(**inputs)
        answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
        
        # Sonuçları kaydet
        st.session_state['qa_log'].append(("💬 " + question, " 👉 "  + answer ))

# Önceki tüm soru-cevapları göster
for i, (user_q, admin_a) in enumerate(reversed(st.session_state['qa_log'])):
    col1, col2 = st.columns([1, 3])  # Sütun genişliklerini belirleyebilirsiniz
    
    with col1:
        st.markdown(f"<p class='question-label'>Question </p>", unsafe_allow_html=True)
        st.markdown(f"<div class='question-textarea'>{user_q}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<p class='question-label'>Answer </p>", unsafe_allow_html=True)
        st.markdown(f"<div class='answer-textarea'>{admin_a}</div>", unsafe_allow_html=True)
    
    st.write("---")

