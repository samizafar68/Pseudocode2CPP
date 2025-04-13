# 🌟 Pseudocode to C++ Code Generator | Transformer from Scratch

A sleek, educational, and fully functional implementation of a Transformer-based Sequence-to-Sequence (Seq2Seq) model that converts **pseudocode ➝ C++ code**.

Built entirely **from scratch in PyTorch** — no pretrained models or external tokenizers — and deployed with a **modern Streamlit app** for real-time interaction.

---

## 📌 Features

✅ Transformer architecture (Encoder-Decoder)  
✅ Custom word-level tokenizer with special token handling (`<sos>`, `<eos>`, `<pad>`)  
✅ Cleaned and realigned SPOC dataset  
✅ Greedy decoding for line-by-line code generation  
✅ Lightweight and extensible implementation  
✅ Interactive **Streamlit** UI with professional styling  
✅ Fully open-source, readable, and easy to learn from  

---

## 🚀 Demo

Try it yourself:

🔗 **Live App**: *[Add your Streamlit link here](https://pseudocode2cpp-pyjrdasgznjeapemzxzxic.streamlit.app/)*  
🔗 **Medium Blog**: *[Add your Medium blog link here](https://medium.com/@sami68/from-pseudocode-to-c-building-a-transformer-model-from-scratch-with-pytorch-0bd068145b32)*  
🔗 **LinkedIn Post**: *[Optional — Add your shareable post if available](https://www.linkedin.com/posts/samiullah68_ai-machinelearning-codegeneration-activity-7317133488426565632-SoOd?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE3bUpsBd9-6QWa_zyHz5Hlv8yJ4AJiW8II)*

---

## 🧠 Concepts Used

### ✨ Transformer (Seq2Seq) – From Scratch
- Encoder-Decoder architecture with `torch.nn.Transformer`
- Positional Encoding to maintain sequence information
- Multi-head self-attention & feed-forward networks
- Greedy decoding with dynamic token generation

### ✨ Custom Tokenizer
- Word-level tokenizer (via regex rules)
- Special token injection: `<sos>`, `<eos>`, `<pad>`
- Clean serialization using `tokenizer.json`

### ✨ Dataset
- 📁 SPOC dataset (Pseudocode & C++ pairs)
- Custom preprocessing for missing/duplicate lines
- Converted TSV ➝ CSV and padded for training

---

## 🧪 Model Architecture

  [Pseudocode]
   ↓
Tokenize + Encode + <sos>/<eos>
   ↓
Embedding + Positional Encoding
   ↓
Transformer Encoder-Decoder
   ↓
Linear + Softmax
   ↓
[Generated C++ Code]
