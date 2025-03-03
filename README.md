# Fluentia - Stuttering Detection and Correction using Machine Learning
## 📌 Project Overview
Fluentia is a speech detection system designed to identify and correct stuttering in individuals. By leveraging **Deep Learning**, we classify different types of stuttering, including **sound repetition, word repetition, silent blocks, interjections, and sound prolongation**. The system utilizes **Bidirectional LSTM (Bi-LSTM) and Gated-Recurrent CNN (GR-CNN)** models to enhance classification accuracy.

## 🛠️ Technologies Used
- **Deep Learning Models:** Bi-LSTM, GR-CNN
- **Speech Processing:** Facebook's **HuBERT-large-ls960-ft** (for speech-to-text conversion)
- **Feature Extraction Techniques:** MFCCs, Delta MFCCs, Delta-delta MFCCs, Wav2Vec
- **Text Processing:** PySpellChecker (for text correction)
- **Text-to-Speech (TTS):** gTTS (Google Text-to-Speech)
- **Speech Recognition:** Google Speech Recognition API
- **Dataset:** SEP-28k (for training and testing)

## 🔍 Methodology
1. **Speech Classification:**
   - Extracted features from speech using **MFCCs, Wav2Vec**, and other techniques.
   - Trained **Bi-LSTM** and **GR-CNN** models to classify stuttering patterns.
   - Achieved **46% accuracy** with Bi-LSTM on Wav2Vec features and **43% accuracy** with GR-CNN.

2. **Speech Correction:**
   - Converted speech to text using **HuBERT-large-ls960-ft**.
   - Applied **PySpellChecker** to correct stuttered words.
   - Used **gTTS** to convert corrected text back into speech.

3. **Practice & Feedback System:**
   - Implemented **Google Speech Recognition** for real-time speech practice.
   - Developed a **comparison system** where incorrect words are highlighted in red, prompting users to repeat them until spoken correctly.

## 📊 Results
- **Bi-LSTM Model:**
  - **Wav2Vec features:** 46% accuracy
  - **MFCCs features:** 31% accuracy
  - **Concatenated MFCCs features:** 33% accuracy
- **GR-CNN Model:**
  - **Wav2Vec features:** 43% accuracy
  - **MFCCs features:** 38% accuracy

## 🎯 Problem Statement
Stuttering is a **neuro-developmental speech disorder** affecting the normal flow of speech. It often includes involuntary **repetitions, prolongations, and silent blocks**, sometimes accompanied by physical behaviors like **head nodding, lip tremors, or rapid eye blinking**. Fluentia aims to improve speech fluency by providing **detection, correction, and real-time practice** to individuals with stuttering disorders.

## 🤝 Contributing
Contributions are welcome! If you'd like to enhance this project, feel free to submit a pull request or open an issue.



