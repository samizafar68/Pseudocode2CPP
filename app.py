import streamlit as st
import torch
import json
import re

# --- Page Configuration ---
st.set_page_config(page_title="PseudoToC++", page_icon="💻", layout="centered")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
        text-align: center;
    }
    textarea {
        font-family: 'Courier New', monospace;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #004999;
    }
    .stCodeBlock {
        background-color: #1e1e1e !important;
        color: #dcdcdc !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Tokenizer from JSON ---
class CustomTokenizer:
    def __init__(self):
        with open("tokenizer.json", "r") as f:
            tokenizer_dict = json.load(f)
        self.word2idx = tokenizer_dict["word2idx"]
        self.idx2word = {int(k): v for k, v in tokenizer_dict["idx2word"].items()}
        self.vocab_size = tokenizer_dict["vocab_size"]
        self.special_tokens = tokenizer_dict["special_tokens"]

    def tokenize(self, text):
        return re.findall(r'\w+|[^\w\s]', text.lower())

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.word2idx.get(token, self.word2idx["<pad>"]) for token in tokens]

    def decode(self, token_ids):
        tokens = [self.idx2word.get(idx, "<unk>") for idx in token_ids]
        return " ".join(tokens)

# Load tokenizer
tokenizer = CustomTokenizer()
SOS_TOKEN_ID = tokenizer.word2idx["<sos>"]
EOS_TOKEN_ID = tokenizer.word2idx["<eos>"]
PAD_TOKEN_ID = tokenizer.word2idx["<pad>"]

# --- Load Model ---
class Transformer(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dff,
            batch_first=True
        )
        self.fc_out = torch.nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.pos_encoding(self.embedding(src))
        tgt_emb = self.pos_encoding(self.embedding(tgt))
        src_padding_mask = (src == PAD_TOKEN_ID)
        tgt_padding_mask = (tgt == PAD_TOKEN_ID)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        out = self.transformer(
            src_emb, tgt_emb,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            tgt_mask=tgt_mask
        )
        return self.fc_out(out)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(4, 256, 4, 1024, tokenizer.vocab_size).to(device)
model.load_state_dict(torch.load("transformer_model.pth", map_location=device))
model.eval()

# --- Code Generation ---
def generate_code(model, pseudocode, max_len=100):
    model.eval()
    with torch.no_grad():
        pseudocode_lines = pseudocode.strip().split('\n')
        generated_code_lines = []

        for line in pseudocode_lines:
            src_tokens = [SOS_TOKEN_ID] + tokenizer.encode(line) + [EOS_TOKEN_ID]
            src = torch.tensor([src_tokens]).to(device)
            tgt = torch.tensor([[SOS_TOKEN_ID]]).to(device)

            for _ in range(max_len):
                output = model(src, tgt)
                next_token = output[:, -1, :].argmax(dim=-1).item()
                if next_token == EOS_TOKEN_ID:
                    break
                tgt = torch.cat([tgt, torch.tensor([[next_token]]).to(device)], dim=1)

            decoded_tokens = tokenizer.decode(tgt[0].tolist())
            decoded_tokens = decoded_tokens.replace("<sos>", "").replace("<eos>", "").strip()
            generated_code_lines.append(decoded_tokens)

        return "\n".join(generated_code_lines)

# --- UI Layout ---
st.title("💡 Pseudocode ➜ C++ Code Generator")
st.markdown("Enter your **pseudocode** and click **Generate** to get C++ code.")

pseudocode = st.text_area("📝 Pseudocode Input", height=200, placeholder="Example:\nSet total to 0\nRepeat 10 times:\n  Add 1 to total")

if st.button("🚀 Generate Code"):
    if pseudocode.strip():
        with st.spinner("Generating..."):
            generated_code = generate_code(model, pseudocode)
        st.subheader("✅ Generated C++ Code")
        st.code(generated_code, language="cpp")
    else:
        st.warning("⚠️ Please enter some pseudocode.")

